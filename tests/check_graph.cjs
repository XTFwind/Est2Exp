// Optional acceptance check: node tests/check_graph.cjs <graph directory> [...]
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {pathToFileURL} = require('node:url');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');

(async () => {
  const browser = await chromium.launch({headless:true, executablePath:process.env.BROWSER_PATH || undefined});
  try {
    const page = await browser.newPage({viewport:{width:1440,height:900},offline:true});
    const errors=[];
    page.on('pageerror', error=>errors.push(error.message));
    page.on('request', request=>{if (/^https?:/.test(request.url())) errors.push(request.url());});
    for (const directory of process.argv.slice(2)) {
      const dag=JSON.parse(fs.readFileSync(path.join(directory,'dag.json'),'utf8'));
      await page.goto(pathToFileURL(path.resolve(directory,'dag.html')).href);
      const actual=await page.evaluate(()=>{
        const svg=document.querySelector('#graph'), box=svg.viewBox.baseVal;
        const groups=[...svg.querySelectorAll('.node')];
        const inside=(x,y)=>x>=box.x&&x<=box.x+box.width&&y>=box.y&&y<=box.y+box.height;
        const clipped=groups.filter(g=>{
          const r=g.getBBox(), t=g.transform.baseVal.getItem(0).matrix;
          return !inside(t.e+r.x,t.f+r.y)||!inside(t.e+r.x+r.width,t.f+r.y+r.height);
        });
        return {ids:groups.map(g=>g.dataset.id),clipped:clipped.map(g=>g.dataset.id),
          edges:[...svg.querySelectorAll('.edge')].map(e=>JSON.stringify([e.dataset.source,e.dataset.target,e.dataset.kind]))};
      });
      assert.deepEqual(actual.ids.sort(),dag.nodes.map(n=>n.id).sort());
      assert.deepEqual(actual.edges.sort(),dag.edges.map(e=>JSON.stringify([e.source,e.target,e.kind])).sort());
      assert.deepEqual(actual.clipped,[]);
      const first=dag.nodes.find(n=>n.kind==='forward'&&n.replica===0&&n.microbatch===0);
      const next=new Map(actual.ids.map(id=>[id,[]]));
      for(const serialized of actual.edges){const [a,b]=JSON.parse(serialized);next.get(a).push(b);}
      const seen=new Set([first.id]), queue=[first.id];
      for(let i=0;i<queue.length;i++)for(const id of next.get(queue[i])){
        if(!seen.has(id)){seen.add(id);queue.push(id);}
      }
      assert(dag.nodes.some(n=>['all_reduce','all_gather'].includes(n.kind)&&seen.has(n.id)));
      const comm=dag.nodes.find(n=>n.kind==='pp_send');
      const event=page.getByRole('button',{name:comm.id,exact:true});
      await event.scrollIntoViewIfNeeded();await event.click();
      assert.equal(await page.locator('#nodeTitle').textContent(),comm.id);
      assert.match(await page.locator('#properties').textContent(),/rounds/);
      await page.locator('#close').click();await page.locator('#critical').check();
      await page.locator('#fit').click();
      assert(await page.locator('#viewport').evaluate(e=>e.scrollWidth<=e.clientWidth+2 && e.scrollHeight<=e.clientHeight+2));
      await page.locator('#reset').click();await page.locator('#in').click();
      assert.equal(await page.locator('#scale').textContent(),'130%');
      await page.locator('#out').click();await page.locator('#critical').uncheck();
      await page.locator('#viewport').evaluate(e=>e.scrollTo(0,0));
      if(process.env.SCREENSHOT_DIR) {
        fs.mkdirSync(process.env.SCREENSHOT_DIR,{recursive:true});
        await page.screenshot({path:path.join(process.env.SCREENSHOT_DIR,path.basename(directory)+'.png')});
      }
      console.log(path.basename(directory)+': '+actual.ids.length+' nodes, '+actual.edges.length+' dependencies, full offline graph verified');
    }
    assert.deepEqual(errors,[]);
  } finally { await browser.close(); }
})().catch(error=>{ console.error(error); process.exitCode=1; });

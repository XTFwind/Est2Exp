"""Small arithmetic language for evaluating a model's ranking programs."""

import ast
import math
import operator

FEATURES = (
    "pp", "dp", "microbatches", "virtual_stages", "schedule", "dp_sync",
    "dp_algorithm", "server_count", "affinity_group_count",
    "max_dp_group_size", "max_dp_servers", "max_dp_affinity_groups", "cross_group_pp_boundaries",
    "pending_microbatch_hint",
)
BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
COMPARE = {
    ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
    ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge,
}
UNARY = {ast.Not: operator.not_, ast.USub: operator.neg, ast.UAdd: operator.pos}
FUNCTIONS = {"abs": abs, "min": min, "max": max}
EXPRESSIONS = (
    ast.Name, ast.Constant, ast.Subscript, ast.BinOp, ast.UnaryOp, ast.Compare,
    ast.IfExp, ast.BoolOp, ast.Call, ast.Load, ast.Store, ast.And, ast.Or,
    *BINARY, *COMPARE, *UNARY,
)


class ScoringProgram:
    def __init__(self, source, strategy_labels=()):
        body = ast.parse(source).body
        if len(body) != 1 or not isinstance(body[0], ast.FunctionDef):
            raise ValueError("Supply one score_strategy(strategy) function")
        function = body[0]
        args = function.args
        if (function.name != "score_strategy" or function.decorator_list
                or len(args.args) != 1 or args.args[0].arg != "strategy" or args.defaults
                or args.kwonlyargs or args.posonlyargs or args.vararg or args.kwarg):
            raise ValueError("Expected score_strategy(strategy)")
        if not isinstance(function.body[-1], ast.Return):
            raise ValueError("End the function with a numeric return expression")
        self.statements = []
        for statement in function.body[:-1]:
            if (not isinstance(statement, ast.Assign) or len(statement.targets) != 1
                    or not isinstance(statement.targets[0], ast.Name)
                    or statement.targets[0].id in {"strategy", *FUNCTIONS}):
                raise ValueError("Only local arithmetic assignments are supported before return")
            self.statements.append((statement.targets[0].id, statement.value))
        self.result = function.body[-1].value
        if self.result is None:
            raise ValueError("Return a numeric score")
        names = set()
        for name, expression in [*self.statements, (None, self.result)]:
            for node in ast.walk(expression):
                if not isinstance(node, EXPRESSIONS):
                    raise ValueError(f"Unsupported scoring expression: {type(node).__name__}")
                if isinstance(node, ast.Name) and node.id not in names | {"strategy", *FUNCTIONS}:
                    raise ValueError(f"Unknown scoring variable: {node.id}")
                if isinstance(node, ast.Subscript) and not (
                    isinstance(node.value, ast.Name) and node.value.id == "strategy"
                    and isinstance(node.slice, ast.Constant) and node.slice.value in FEATURES
                ):
                    raise ValueError("Read only the listed strategy features")
                if isinstance(node, ast.Call) and not (
                    isinstance(node.func, ast.Name) and node.func.id in FUNCTIONS and not node.keywords
                ):
                    raise ValueError("Only abs, min and max calls are supported")
                if isinstance(node, ast.Constant) and type(node.value) not in (bool, int, float, str):
                    raise ValueError("Unsupported literal in scoring program")
                if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                        and node.value not in FEATURES and node.value not in strategy_labels):
                    raise ValueError("Use only public feature and strategy names as string literals")
            names.add(name)
        # Keep only the interpreted program, including no free text from provider responses.
        self.source = "def score_strategy(strategy):\n" + "\n".join(
            "    " + ast.unparse(statement) for statement in function.body
        )

    def __call__(self, features):
        variables = {}

        def value(node):
            if isinstance(node, ast.Constant):
                return node.value
            if isinstance(node, ast.Name):
                return variables[node.id]
            if isinstance(node, ast.Subscript):
                return features[node.slice.value]
            if isinstance(node, ast.BinOp):
                return BINARY[type(node.op)](float(value(node.left)), float(value(node.right)))
            if isinstance(node, ast.UnaryOp):
                return UNARY[type(node.op)](value(node.operand))
            if isinstance(node, ast.IfExp):
                return value(node.body if value(node.test) else node.orelse)
            if isinstance(node, ast.BoolOp):
                for operand in node.values:
                    result = value(operand)
                    if bool(result) == isinstance(node.op, ast.Or):
                        break
                return result
            if isinstance(node, ast.Compare):
                left = value(node.left)
                for comparison, other in zip(node.ops, node.comparators):
                    right = value(other)
                    if not COMPARE[type(comparison)](left, right):
                        return False
                    left = right
                return True
            return FUNCTIONS[node.func.id](*(float(value(arg)) for arg in node.args))

        for name, expression in self.statements:
            variables[name] = value(expression)
        score = float(value(self.result))
        if not math.isfinite(score):
            raise ValueError("Return a finite score")
        return score

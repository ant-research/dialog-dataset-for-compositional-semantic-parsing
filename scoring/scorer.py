"""Supports Function Expression parsing and scoring.

Usage:

python3 scorer.py reference.jsonl prediction.jsonl -N eval_norm.json

This will print utterance level accuracy.
"""

from scoring.function import Function
from typing import Tuple, List, Dict

import argparse
import json
import re
import sys


Argument = Function.Argument


class FunctionExpression(object):
    """Class for Function Expression parsing.

    Args:
        string_expr: the string to be parsed.

    Attributes:
        function: parsing result as a Function. None means invalid.
    """

    def __init__(self, string_expr: str = None, function: Function = None):
        super().__init__()
        if string_expr is not None:
            self.function = self.parse(string_expr)
        elif function is not None:
            self.function = function
        else:
            self.function = None

    @property
    def is_valid(self) -> bool:
        return self.function is not None

    def _parse_func(self, input) -> Tuple[Function, str]:
        """Parse a Function message from the beginning of input."""
        m = re.match(r"([\w\]\[<>&]+)(?:/(\w+))?\s*\(\s*", input)
        if m is None:
            print(f"Cannot match function from input: {input}")
            return None, None
        rest = input[m.span()[1] :]
        domain, intent = m.groups()
        if intent is None:
            name = domain
            domain = None
        else:
            name = f"{domain}/{intent}"

        args, rest = self._parse_args(rest)
        if args is None:
            return None, None

        func = Function(name=name, args=args)
        if domain is not None:
            func.intent_info.domain = domain
            func.intent_info.intent = intent

        return func, rest

    def _parse_args(self, input) -> List[Argument]:
        """Parses function's argument list from the beginning of input."""
        if not input:
            return None, None
        rest = input
        args = []
        if rest[0] == ")":
            return args, rest[1:].lstrip()

        named_arg_only = False
        while rest:
            arg, rest = self._parse_arg(rest, named_arg_only)
            if arg is None:
                print(f"cannot get arg from '{rest}")
                return None, None
            if arg.name:
                named_arg_only = True

            args.append(arg)
            if not rest:
                return None, None
            if rest[0] == ",":
                rest = rest[1:].lstrip()
            elif rest[0] == ")":
                return args, rest[1:].lstrip()
            else:
                print(f"unexpected char {rest[0]}, expecting , or ), in '{input}'")
                return None, None
        return None, None

    def _parse_arg(self, input, named_arg_only: bool) -> Tuple[Argument, str]:
        """Parses a single argument from the beginning of input."""
        m = re.match(r"^(\w+)\s*=\s*(.*)$", input)
        if m is not None:
            name, rest = m.groups()
        elif not named_arg_only:
            name, rest = "", input
        else:
            print(f"cannot match arg from {input}")
            return None, None

        arg = Function.Argument(
            name=name,
        )
        arg, rest = self._parse_arg_value(arg, rest)
        if arg is None:
            return None, None
        return arg, rest

    def _parse_arg_value(self, arg, input) -> Tuple[Argument, str]:
        """Parses argument value from the beginning of input."""
        m = re.match(r"'(.*?)'\s*(?:{(\w+)})?\s*", input)
        if m is None:
            m = re.match(r'"(.*?)"\s*(?:{(\w+)})?\s*', input)

        if m is not None:
            val, vtype = m.groups()
            arg.literal.value = val
            if vtype is not None:
                arg.literal.type = vtype

            return arg, input[m.span()[1] :]

        m = re.match(r"(\w+)/(\w+)/(\w+(?:\.[^\s,\)]+)?)\s*", input)
        if m is not None:
            domain, intent, slot_spec = m.groups()
            arg.slot_ref.intent_info.domain = domain
            arg.slot_ref.intent_info.intent = intent
            arg.slot_ref.slot_spec = slot_spec
            return arg, input[m.span()[1] :]

        func, rest = self._parse_func(input)
        if func is None or not rest:
            print(f"failed to parse function from '{input}'")
            return None, None

        if func.HasField("intent_info"):
            arg.slot_ref.function.CopyFrom(func)
            if rest[0] == "/":  # followed by slot spec
                m = re.match(r"/(\w+(?:\.[^\s,)]+)?)\s*", rest)
                if m is None:
                    print(f"failed to match slot spec @'{rest}")
                    return None, None
                slot_spec, rest = m.group(1), rest[m.span()[1] :]
                arg.slot_ref.slot_spec = slot_spec
            elif rest[0] == "," or rest[0] == ")":  # splot spec omitted
                arg.slot_ref.slot_spec = "__"
            else:  # invalid
                print(f"unexpected REST: {rest}")
                return None, None
        else:
            arg.function.CopyFrom(func)

        return arg, rest

    def parse(self, input) -> Function:
        """Parses the entire input into a Function message.

        A successful parse must consume the entire string.

        Args:
            input: text to be parsed.

        Returns:
            A Function message if successful, otherwise None.
        """
        input = input.replace("\n", " ").strip()
        function, rest = self._parse_func(input)
        if function is None or rest:
            return None
        else:
            self.function = function
            return function

    def normalize(self, sort_args: bool = True) -> str:
        """Outputs normalized form of the function expression."""
        if self.function is None:
            return None
        elif self.function.normed_expr:
            return self.function.normed_expr

        if sort_args:
            self.function.sort_named_args()

        self.function.normed_expr = str(self.function)
        return self.function.normed_expr


class Scorer(object):
    CASERS = {"upper": lambda x: x.upper(), "lower": lambda x: x.lower(), "none": None}

    def __init__(self, norm_file: str = None, sort_args: bool = True):
        self._load_norm_rules(norm_file)
        self._sort_args = sort_args

    def _load_norm_rules(self, norm_file: str):
        self._caser = None
        self._norm_rules = []
        if norm_file is not None:
            with open(norm_file, "r") as fh:
                rules = json.load(fh)
                casing = rules.get("casing", "lower").lower()
                if casing not in self.CASERS:
                    raise ValueError(f"unknown casing {casing}.")

                self._caser = self.CASERS.get(casing)

                for line in rules.get("filters", []):
                    pattern = line.strip()
                    rgx = re.compile(pattern)
                    self._norm_rules.append((rgx, ""))

                for ptn, repl in rules.get("subs", {}).items():
                    rgx = re.compile(ptn)
                    self._norm_rules.append((rgx, repl))

                print(f"loaded {len(self._norm_rules)} norm rules.")

    def read(self, fname: str, strict: bool = False) -> Dict[str, FunctionExpression]:
        result = {}
        with open(fname, "r", encoding="utf-8") as fh:
            for line in fh:
                data = json.loads(line)
                target = data["target"]
                if self._caser is not None:
                    target = self._caser(target)
                if self._norm_rules:
                    for rgx, repl in self._norm_rules:
                        target = re.sub(rgx, repl, target)
                fe = FunctionExpression(target or None)
                if strict and not fe.is_valid:
                    raise ValueError(f"failed to parse {target}")

                result[data["id"]] = fe

        return result

    def score(self, ref_file: str, pred_file: str) -> None:
        num_corr = 0
        refs = self.read(ref_file, True)
        pred = self.read(pred_file, False)
        for id, fe in pred.items():
            if id not in refs:
                raise ValueError(f"cannot find reference for {id}.")

            ref = refs[id]
            if fe.is_valid and fe.normalize(sort_args=self._sort_args) == ref.normalize(
                sort_args=self._sort_args
            ):
                num_corr += 1

        num_pred = len(pred)
        print(f"accuracy = {num_corr / num_pred:.4f} ({num_corr}/{num_pred}).")


def main(args: argparse.Namespace) -> None:
    scorer = Scorer(args.norm, not args.order_sensitive)
    scorer.score(args.reference, args.prediction)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="scorer")
    parser.add_argument("reference", help="reference file in JSONL format.")
    parser.add_argument("prediction", help="prediction file in JSONL format.")
    parser.add_argument(
        "-N",
        "--norm",
        help="optional file that defines casing and normalization rules.",
    )
    parser.add_argument(
        "--order-sensitive",
        action="store_true",
        help="if true, will not sort named arguments.",
    )
    args = parser.parse_args()
    sys.exit(main(args))

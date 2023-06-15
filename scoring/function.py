from copy import deepcopy


class Message(object):
    """Simulates protobuf message.

    This class is designed for keeping internal and public version
    of the scorer compatible. The internal version is using protobuf,
    but the public version does not, in order to avoid extra dependencies.
    """

    def HasField(self, field: str) -> bool:
        if hasattr(self, f"_{field}"):
            return getattr(self, f"_{field}") is not None
        else:
            return getattr(self, field, None) is not None

    def CopyFrom(self, other) -> None:
        assert self.__class__ == other.__class__
        self.__dict__ = deepcopy(other.__dict__)


class Function(Message):
    """Function message."""

    class IntentInfo(Message):
        def __init__(self, domain: str = None, intent: str = None):
            self.domain = domain
            self.intent = intent

        def __str__(self):
            return f"{self.domain}/{self.intent}"

    class SlotRef(Message):
        """SlotRef message."""

        def __init__(self, slot_spec=None, intent_info=None, function=None):
            self.slot_spec = slot_spec
            self._intent_info = intent_info
            self._function = function

        @property
        def intent_info(self):
            if self._intent_info is None:
                self._intent_info = Function.IntentInfo()
                self._function = None
            return self._intent_info

        @intent_info.setter
        def intent_info(self, v):
            self._intent_info = v
            self._function = None

        @property
        def function(self):
            if self._function is None:
                self._function = Function()
                self._intent_info = None
            return self._function

        @function.setter
        def function(self, v):
            self._function = v
            self._intent_info = None

        def __str__(self):
            if self._intent_info is not None:
                return f"{self._intent_info}/{self.slot_spec}"
            elif self._function is not None:
                return f"{self._function}/{self.slot_spec}"
            else:
                return "NULL"

    class Literal(Message):
        """Literal message."""

        def __init__(self, value=None, type=None):
            self.value = value
            self.type = type

        def __str__(self):
            if self.type is None:
                return f'"{self.value}"'
            else:
                return f'"{self.value}"{{{self.type}}}'

    class Argument(Message):
        """Argument message."""

        def __init__(self, name=None, function=None, slot_ref=None, literal=None):
            super().__init__()
            self.name = name
            self._function = function
            self._slot_ref = slot_ref
            self._literal = literal

        @property
        def function(self):
            if self._function is None:
                self._function = Function()
                self._slot_ref, self._literal = None, None
            return self._function

        @function.setter
        def function(self, v):
            self._function = v
            self._slot_ref, self._literal = None, None

        @property
        def slot_ref(self):
            if self._slot_ref is None:
                self._slot_ref = Function.SlotRef()
                self._function, self._literal = None, None
            return self._slot_ref

        @slot_ref.setter
        def slot_ref(self, v):
            self._slot_ref = v
            self._function, self._literal = None, None

        @property
        def literal(self):
            if self._literal is None:
                self._literal = Function.Literal()
                self._function, self._slot_ref = None, None
            return self._literal

        @literal.setter
        def literal(self, v):
            self._literal = v
            self._function, self._slot_ref = None, None

        def __str__(self):
            pfx = f"{self.name}=" if self.name else ""
            if self._function is not None:
                return f"{pfx}{self._function}"
            elif self._slot_ref is not None:
                return f"{pfx}{self._slot_ref}"
            elif self._literal is not None:
                return f"{pfx}{self._literal}"
            else:
                return "NULL_ARG"

    # for Function class
    def __init__(self, name=None, args=None, intent_info=None):
        if name is not None:
            self.name = name
        self.args = args or []
        self._intent_info = intent_info
        self.normed_expr = None

    @property
    def intent_info(self):
        if self._intent_info is None:
            self._intent_info = Function.IntentInfo()
        return self._intent_info

    @intent_info.setter
    def intent_info(self, val):
        self._intent_info = val

    def __str__(self):
        return f"{self.name}({','.join([str(a) for a in self.args])})"

    def sort_named_args(self):
        n = len(self.args)
        i = 0
        for i in range(n):
            if self.args[i].name:
                break

        # the rest should all be named arguments, so sort them by name.
        if i < n - 1:
            positional_args = self.args[:i]
            named_args = self.args[i:]
            named_args.sort(key=lambda arg: arg.name)
            self.args = positional_args + named_args

        for arg in self.args:
            if arg._function is not None:
                arg._function.sort_named_args()
            elif arg._slot_ref is not None and arg._slot_ref._function is not None:
                arg._slot_ref._function.sort_named_args()

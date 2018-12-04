__author__ = "Yngve Mardal Moe"
__email__ = "yngve.m.moe@gmail.com"


import difflib


class SubclassRegister:
    """Creates a register instance used to register all subclasses of some base class.

    Use the `SubclassRegister.link` decorator to link a base class with
    the register.

    Example:
    --------
    >>> register = SubclassRegister('car')
    >>> 
    >>> @register.register_base
    >>> class BaseCar:
    >>>     pass
    >>> 
    >>> class SUV(BaseCar):
    >>>     def __init__(self, num_seats):
    >>>         self.num_seats = num_seats
    >>> 
    >>> class Sedan(BaseCar):
    >>>     def __init__(self, num_seats):
    >>>         self.num_seats = num_seats
    >>> 
    >>> print(register.available_classes)
    ('SUV', 'Sedan')
    (SUV, Sedan)
    >>> print(register.get_item('SUV'))
    <class '__main__.SUV'>
    >>> print(register.get_item('sedan'))
    ValueError: sedan is not a valid name for a car. 
    Available cars are (in decreasing similarity):
       * Sedan
       * SUV
    """

    def __init__(self, class_name):
        """

        Arguments:
        ----------
        class_name : str
            The name of the classes we register, e.g. layer or model.
            Used for errors.
        """
        self.class_name = class_name
        self.register = {}
        self.linked_base = None

    @property
    def available_classes(self):
        return tuple(self.register.keys())

    @property
    def linked(self):
        if self.linked_base is None:
            return False
        return True

    def link_base(self, cls):
        """Link a base class to the register. Can be used as a decorator.
        """
        if self.linked:
            raise RuntimeError(
                "Cannot link the same register with two different base classes"
            )

        @classmethod
        def init_subclass(obj):
            name = obj.__name__
            if name in self.register:
                raise ValueError(
                    f"Cannot create two {self.class_name}s with the same name."
                )
            self.register[name] = obj

        self.linked_base = cls
        cls.__init_subclass__ = init_subclass
        return cls

    def skip(self, cls):
        if not self.linked:
            raise RuntimeError(
                "The register must be linked to a base class before a subclass can be skipped."
                )
        if not issubclass(cls, self.linked_base):
            raise ValueError(
                f"{cls.__name__} is not a subclass of {self.linked_base.__name__}"
            )
        del self.register[cls.__name__]
        return cls

    def get_item(self, item):
        if item not in self.register:

            def get_similarity(item_):
                return difflib.SequenceMatcher(None, item, item_).ratio()

            traceback = f"{item} is not a valid name for a {self.class_name}."
            traceback = f"{traceback} \nAvailable {self.class_name}s are (in decreasing similarity):"

            sorted_items = sorted(self.register, key=get_similarity, reverse=True)
            for available in sorted_items:
                traceback = f"{traceback}\n   * {available}"

            raise ValueError(traceback)
        return self.register[item]


if __name__ == "__main__":
    register = SubclassRegister("car")

    @register.link_base
    class BaseCar:
        pass

    class SUV(BaseCar):
        def __init__(self, num_seats):
            self.num_seats = num_seats

    class Sedan(BaseCar):
        def __init__(self, num_seats):
            self.num_seats = num_seats

    @register.skip
    class ToyCar(BaseCar):
        def __init__(self, weight):
            self.weight = weight

    print(register.available_classes)
    print(register.get_item("SUV"))
    print(register.get_item("ToyCar"))

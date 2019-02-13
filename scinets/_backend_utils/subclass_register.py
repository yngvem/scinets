__author__ = "Yngve Mardal Moe"
__email__ = "yngve.m.moe@gmail.com"


import difflib


class BaseRegister:
    def __init__(self):
        self.register = {}

    def get_items_by_similarity(self, item):
        def get_similarity(item_):
            return difflib.SequenceMatcher(None, item.lower(), item_.lower()).ratio()

        return sorted(self.register, key=get_similarity, reverse=True)

    def validate_item_in_register(self, item):
        if item not in self.register:
            traceback = f"{item} is not a valid name for a {self.class_name}."
            traceback = f"{traceback} \nAvailable {self.class_name}s are (in decreasing similarity):"

            sorted_items = self.get_items_by_similarity(item)
            for available in sorted_items:
                traceback = f"{traceback}\n   * {available}"

            raise IndexError(traceback)

    def __getitem__(self, item):
        return self.get_item(item)

    def get_item(self, item):
        self.validate_item_in_register(item)
        return self.register[item]

    def add_item(self, name, item):
        if name in self.register:
            raise ValueError(f"Cannot register two items with the same name")
        self.register[name] = item

    def remove_item(self, name):
        self.validate_item_in_register(item)
        del self.register[name]


class DictionaryRegister(BaseRegister):
    def __init__(self, register):
        super().__init__()
        self.register = register


class SubclassRegister(BaseRegister):
    """Creates a register instance used to register all subclasses of some base class.

    Use the `SubclassRegister.link` decorator to link a base class with
    the register.

    Example:
    --------
    >>> register = SubclassRegister('car')
    >>> 
    >>> @register.link_base
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
        self.linked_base = None
        super().__init__()

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

        old_init_subclass = cls.__init_subclass__

        @classmethod
        def init_subclass(cls_, *args, **kwargs):
            name = cls_.__name__
            if name in self.register:
                raise ValueError(
                    f"Cannot create two {self.class_name}s with the same name."
                )
            self.add_item(name, cls_)
            return old_init_subclass(*args, **kwargs)

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
        self.remove_item(cls.__name__)

        return cls


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

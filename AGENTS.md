# CFGRL

most activity will occur in `ogbench` dir

# THINGS I LIKE

* I prefer very concise docstrings. and concise code. the purpose of the code
  should be clear from reading its contents

if its a dataclass, then
@dataclass class Myclass:
""" Myclass is a dataclass that represents a simple example. """
    name: str # it is better to describe the argument here
    age: int # and here rather than the docstring

* I prefer OOP paterns for different components.
* wrapper/decorator can sometimes be a good choice
* I also like using a config.create(*args, **kwargs) to create components from
  config objects

* don't be afraid to propose new code restructuring and reorganizing if doing
  so benefits maintainability, but do not do so frivolously

**multi file organization**
*	Use feature-based style rather than layer based. Keeps complexity isolated
*	Add a shared/infra or common/ layer for reusable low-level primitives

# THINGS I DON'T LIKE

* excessive nesting. except when necessary, components should be flat. list and
  dict comprehension is fine. jax.tree.map is also fine
* long variable names. this clutters the code. try to keep them short and
  meaningful.
* Long Functions or Methods. multiple responsibilities can be challenging to
  test and maintain. Break down long functions into smaller, single-purpose
  functions to adhere to the Single Responsibility Principle. cyclomatic complexity (CC) should be no more than 8, but 4-5 is better.
* Duplicated Code. keep it dry

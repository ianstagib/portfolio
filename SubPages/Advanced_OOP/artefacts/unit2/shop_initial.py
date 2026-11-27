"""The starting point for the Unit 2 case study: a poorly designed shopping
cart, kept close to the code given in the brief so its problems can be seen
and tested before the refactor in shop_solid.py.

Two small additions make it runnable. The brief's ``Order`` reads
``item.price`` but never defines an item, so ``Item`` is added here. The
brief also asks for discounts, which the original ``Order`` cannot do, so
``apply_discount()`` is added in the same style as ``pay()``: one method
that grows a new branch for every new kind of discount.

Problems, numbered as in the brief:

1. SRP: ``Order`` manages the cart, calculates totals, applies discounts
   and processes payments, so it has several reasons to change.
2. OCP: a new payment method or discount means editing ``Order``.
3. DIP: ``Order`` is tied to the concrete payment methods, which it
   names as strings and handles inline.

Three further problems showed up while testing it:

* ``pay()`` prints instead of returning a result, so the only way to test
  it is to capture standard output, and a caller cannot tell whether a
  payment succeeded.
* An unknown payment type does nothing at all: no error, no result.
* Prices are floats, so totals can pick up binary rounding errors.

>>> order = Order()
>>> order.add_item(Item("Notebook", 0.10))
>>> order.add_item(Item("Pencil", 0.20))
>>> order.calculate_total()
0.30000000000000004
>>> order.pay("credit")
Processing credit card payment...
>>> order.pay("applepay")  # silently ignored
"""


class Item:
    """A product line in the cart. Not in the brief, but ``Order`` needs it."""

    def __init__(self, name, price):
        self.name = name
        self.price = price


class Order:

    def __init__(self):
        self.items = []
        self.discount_type = None
        self.discount_value = 0

    def add_item(self, item):
        self.items.append(item)

    def apply_discount(self, discount_type, value):
        # Added for the brief's discount requirement, in the same style as
        # pay(): every new kind of discount is another branch here.
        self.discount_type = discount_type
        self.discount_value = value

    def calculate_total(self):
        total = sum(item.price for item in self.items)
        if self.discount_type == "percent":
            total = total - total * self.discount_value / 100
        elif self.discount_type == "fixed":
            total = total - self.discount_value
        return total

    def pay(self, payment_type):
        if payment_type == "credit":
            print("Processing credit card payment...")
        elif payment_type == "paypal":
            print("Processing PayPal payment...")
        # Adding a new method requires modifying this class.

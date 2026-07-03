#!/usr/bin/env python3
"""Synthetic DISPATCH-AMBIGUITY tasks on a real language server (pyrefly).

Motivation (docs/real_repo_progress.md "semantic vs textual" / dispatch sections; REPORT.md 3-4):
a method `NAME` is overridden on N>=8 classes, so `grep 'def NAME'` returns N candidates and cannot
say WHICH one binds -- that depends on the receiver's *static type*, which grep cannot compute but a
type-aware go-to-definition (pyrefly LSP) can. Each task plants exactly ONE bug, in the single
override that binds for a statically-typed receiver in `pkg/app.py`. The agent must localize that one
override among N and fix a one-line bug.

`build_tasks(tmp_root)` materializes K=15 self-contained on-disk repos and returns a task dict each.
Run this module directly for GATE 1 (no model): it checks, per task, that the test fails at base,
passes after the gold fix, that grep sees >=8 `def NAME`, and that pyrefly's receiver-aware goto
resolves to the RIGHT (buggy) override file rather than a sibling.

  .venv-streams.system/bin/python scripts/realbench/dispatch_tasks.py
"""
from __future__ import annotations
import os
import re
import sys
import ast
import shutil
import tempfile
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ---------------------------------------------------------------------------
# Task source templates. Each package overrides one method `NAME` on 8 to 15
# classes spread across 2 to 3 files; app.py constructs a statically-typed
# receiver of ONE class and calls x.NAME(...); exactly that class's override
# carries the bug.
# (Regular strings, NOT f-strings: the code below contains literal braces.)
# ---------------------------------------------------------------------------

# ===== Task A: codecs, method `serialize` (receiver JsonHandler) =====
A_BASE = '''\
class Handler:
    """Base codec: turn a dict into its serialized string form."""

    name = "base"

    def serialize(self, data: dict) -> str:
        raise NotImplementedError
'''

A_TEXT = '''\
import json

from pkg.base import Handler


class JsonHandler(Handler):
    name = "json"

    def serialize(self, data: dict) -> str:
        return json.dumps(data)


class XmlHandler(Handler):
    name = "xml"

    def serialize(self, data: dict) -> str:
        body = "".join("<%s>%s</%s>" % (k, v, k) for k, v in sorted(data.items()))
        return "<root>%s</root>" % body


class YamlHandler(Handler):
    name = "yaml"

    def serialize(self, data: dict) -> str:
        return "\\n".join("%s: %s" % (k, v) for k, v in sorted(data.items()))
'''

A_BINARY = '''\
import base64
import binascii

from pkg.base import Handler


class Base64Handler(Handler):
    name = "base64"

    def serialize(self, data: dict) -> str:
        raw = "&".join("%s=%s" % (k, v) for k, v in sorted(data.items())).encode()
        return base64.b64encode(raw).decode()


class HexHandler(Handler):
    name = "hex"

    def serialize(self, data: dict) -> str:
        raw = "&".join("%s=%s" % (k, v) for k, v in sorted(data.items())).encode()
        return binascii.hexlify(raw).decode()


class PickleHandler(Handler):
    name = "pickle"

    def serialize(self, data: dict) -> str:
        return "|".join("%s:%r" % (k, v) for k, v in sorted(data.items()))
'''

A_TABULAR = '''\
from pkg.base import Handler


class CsvHandler(Handler):
    name = "csv"

    def serialize(self, data: dict) -> str:
        keys = sorted(data)
        return ",".join(keys) + "\\n" + ",".join(str(data[k]) for k in keys)


class TsvHandler(Handler):
    name = "tsv"

    def serialize(self, data: dict) -> str:
        keys = sorted(data)
        return "\\t".join(keys) + "\\n" + "\\t".join(str(data[k]) for k in keys)


class IniHandler(Handler):
    name = "ini"

    def serialize(self, data: dict) -> str:
        return "\\n".join("%s = %s" % (k, v) for k, v in sorted(data.items()))
'''

A_APP = '''\
from pkg.handlers.text import JsonHandler


def run(x: JsonHandler, data: dict) -> str:
    """Serialize `data` with the given handler and return the string form."""
    return x.serialize(data)
'''

A_TEST = '''\
from pkg.app import run
from pkg.handlers.text import JsonHandler


def test_json_serialize_sorts_keys():
    # canonical JSON emits object keys in sorted order
    assert run(JsonHandler(), {"b": 2, "a": 1}) == '{"a": 1, "b": 2}'


if __name__ == "__main__":
    test_json_serialize_sorts_keys()
    print("OK")
'''

# ===== Task B: field validators, method `validate` (receiver EmailField) =====
B_BASE = '''\
class Field:
    """Base validator: return True iff `value` is well-formed for this field."""

    name = "base"

    def validate(self, value: str) -> bool:
        raise NotImplementedError
'''

B_SCALAR = '''\
from pkg.base import Field


class IntField(Field):
    name = "int"

    def validate(self, value: str) -> bool:
        return value.lstrip("-").isdigit()


class FloatField(Field):
    name = "float"

    def validate(self, value: str) -> bool:
        try:
            float(value)
            return True
        except ValueError:
            return False


class BoolField(Field):
    name = "bool"

    def validate(self, value: str) -> bool:
        return value in ("true", "false")
'''

B_TEXT = '''\
from pkg.base import Field


class StrField(Field):
    name = "str"

    def validate(self, value: str) -> bool:
        return len(value) > 0


class EmailField(Field):
    name = "email"

    def validate(self, value: str) -> bool:
        return "@" in value


class SlugField(Field):
    name = "slug"

    def validate(self, value: str) -> bool:
        return len(value) > 0 and all(c.isalnum() or c == "-" for c in value)
'''

B_NET = '''\
from pkg.base import Field


class UrlField(Field):
    name = "url"

    def validate(self, value: str) -> bool:
        return value.startswith(("http://", "https://"))


class IpField(Field):
    name = "ip"

    def validate(self, value: str) -> bool:
        parts = value.split(".")
        return len(parts) == 4 and all(p.isdigit() and 0 <= int(p) <= 255 for p in parts)


class UuidField(Field):
    name = "uuid"

    def validate(self, value: str) -> bool:
        hexpart = value.replace("-", "").lower()
        return len(hexpart) == 32 and all(c in "0123456789abcdef" for c in hexpart)
'''

B_APP = '''\
from pkg.fields.text import EmailField


def run(x: EmailField, value: str) -> bool:
    """Validate `value` with the given field and return the verdict."""
    return x.validate(value)
'''

B_TEST = '''\
from pkg.app import run
from pkg.fields.text import EmailField


def test_email_requires_domain_dot():
    assert run(EmailField(), "user@example.com") is True
    assert run(EmailField(), "user@localhost") is False


if __name__ == "__main__":
    test_email_requires_domain_dot()
    print("OK")
'''

# ===== Task C: expression nodes, method `to_str` (receiver MulNode) =====
C_BASE = '''\
class Node:
    """Base expression node: render itself to a string."""

    def to_str(self) -> str:
        raise NotImplementedError
'''

C_ATOM = '''\
from pkg.base import Node


class NumNode(Node):
    def __init__(self, value: int) -> None:
        self.value = value

    def to_str(self) -> str:
        return str(self.value)


class VarNode(Node):
    def __init__(self, name: str) -> None:
        self.vname = name

    def to_str(self) -> str:
        return self.vname
'''

C_ARITH = '''\
from pkg.base import Node


class AddNode(Node):
    def __init__(self, left: Node, right: Node) -> None:
        self.left = left
        self.right = right

    def to_str(self) -> str:
        return "(%s + %s)" % (self.left.to_str(), self.right.to_str())


class SubNode(Node):
    def __init__(self, left: Node, right: Node) -> None:
        self.left = left
        self.right = right

    def to_str(self) -> str:
        return "(%s - %s)" % (self.left.to_str(), self.right.to_str())


class MulNode(Node):
    def __init__(self, left: Node, right: Node) -> None:
        self.left = left
        self.right = right

    def to_str(self) -> str:
        return "(%s + %s)" % (self.left.to_str(), self.right.to_str())


class DivNode(Node):
    def __init__(self, left: Node, right: Node) -> None:
        self.left = left
        self.right = right

    def to_str(self) -> str:
        return "(%s / %s)" % (self.left.to_str(), self.right.to_str())
'''

C_MISC = '''\
from pkg.base import Node


class PowNode(Node):
    def __init__(self, base: Node, exp: Node) -> None:
        self.base = base
        self.exp = exp

    def to_str(self) -> str:
        return "(%s ** %s)" % (self.base.to_str(), self.exp.to_str())
'''

C_APP = '''\
from pkg.nodes.arith import MulNode


def run(x: MulNode) -> str:
    """Render the expression node to a string."""
    return x.to_str()
'''

C_TEST = '''\
from pkg.app import run
from pkg.nodes.arith import MulNode
from pkg.nodes.atom import NumNode


def test_mul_renders_product():
    assert run(MulNode(NumNode(6), NumNode(7))) == "(6 * 7)"


if __name__ == "__main__":
    test_mul_renders_product()
    print("OK")
'''


# ===== Task D: text encoders, method `encode` (receiver ReverseEncoder), N=8 =====
# Small ~2-line bodies; bug = wrong slice-step operator.
D_BASE = '''\
class Encoder:
    """Base text encoder: transform `text` into its encoded form."""

    name = "base"

    def encode(self, text: str) -> str:
        raise NotImplementedError
'''

D_SIMPLE = '''\
from pkg.base import Encoder


class UpperEncoder(Encoder):
    name = "upper"

    def encode(self, text: str) -> str:
        return text.upper()


class LowerEncoder(Encoder):
    name = "lower"

    def encode(self, text: str) -> str:
        return text.lower()


class ReverseEncoder(Encoder):
    name = "reverse"

    def encode(self, text: str) -> str:
        return text[::-2]
'''

D_REPEAT = '''\
from pkg.base import Encoder


class DoubleEncoder(Encoder):
    name = "double"

    def encode(self, text: str) -> str:
        return text * 2


class SpaceEncoder(Encoder):
    name = "space"

    def encode(self, text: str) -> str:
        return " ".join(text)


class StripEncoder(Encoder):
    name = "strip"

    def encode(self, text: str) -> str:
        return text.strip()
'''

D_COUNT = '''\
from pkg.base import Encoder


class LenEncoder(Encoder):
    name = "len"

    def encode(self, text: str) -> str:
        return str(len(text))


class HeadEncoder(Encoder):
    name = "head"

    def encode(self, text: str) -> str:
        return text[:1]
'''

D_APP = '''\
from pkg.codecs.simple import ReverseEncoder


def run(x: ReverseEncoder, text: str) -> str:
    """Encode `text` with the given encoder and return the result."""
    return x.encode(text)
'''

D_TEST = '''\
from pkg.app import run
from pkg.codecs.simple import ReverseEncoder


def test_reverse_encodes_backwards():
    assert run(ReverseEncoder(), "abcd") == "dcba"


if __name__ == "__main__":
    test_reverse_encodes_backwards()
    print("OK")
'''

# ===== Task E: UI elements, method `render` (receiver CheckboxElement), N=12 =====
# Small bodies; bug = swapped ternary branch.
E_BASE = '''\
class Element:
    """Base UI element: render itself to a string."""

    def render(self) -> str:
        raise NotImplementedError
'''

E_INLINE = '''\
from pkg.base import Element


class BoldElement(Element):
    def __init__(self, text: str) -> None:
        self.text = text

    def render(self) -> str:
        return "**" + self.text + "**"


class ItalicElement(Element):
    def __init__(self, text: str) -> None:
        self.text = text

    def render(self) -> str:
        return "_" + self.text + "_"


class CodeElement(Element):
    def __init__(self, text: str) -> None:
        self.text = text

    def render(self) -> str:
        return "`" + self.text + "`"


class LinkElement(Element):
    def __init__(self, text: str, href: str) -> None:
        self.text = text
        self.href = href

    def render(self) -> str:
        return "[" + self.text + "](" + self.href + ")"
'''

E_BLOCK = '''\
from pkg.base import Element


class HeadingElement(Element):
    def __init__(self, text: str) -> None:
        self.text = text

    def render(self) -> str:
        return "# " + self.text


class QuoteElement(Element):
    def __init__(self, text: str) -> None:
        self.text = text

    def render(self) -> str:
        return "> " + self.text


class BulletElement(Element):
    def __init__(self, text: str) -> None:
        self.text = text

    def render(self) -> str:
        return "- " + self.text


class RuleElement(Element):
    def __init__(self) -> None:
        self.text = ""

    def render(self) -> str:
        return "---"
'''

E_STATE = '''\
from pkg.base import Element


class CheckboxElement(Element):
    def __init__(self, label: str, checked: bool) -> None:
        self.label = label
        self.checked = checked

    def render(self) -> str:
        mark = "[ ]" if self.checked else "[x]"
        return mark + " " + self.label


class ToggleElement(Element):
    def __init__(self, label: str, on: bool) -> None:
        self.label = label
        self.on = on

    def render(self) -> str:
        state = "ON" if self.on else "OFF"
        return self.label + ": " + state


class BadgeElement(Element):
    def __init__(self, label: str, active: bool) -> None:
        self.label = label
        self.active = active

    def render(self) -> str:
        inner = self.label if self.active else " "
        return "(" + inner + ")"


class StatusElement(Element):
    def __init__(self, label: str, ok: bool) -> None:
        self.label = label
        self.ok = ok

    def render(self) -> str:
        verdict = "PASS" if self.ok else "FAIL"
        return self.label + " " + verdict
'''

E_APP = '''\
from pkg.ui.state import CheckboxElement


def run(x: CheckboxElement) -> str:
    """Render the given UI element to a string."""
    return x.render()
'''

E_TEST = '''\
from pkg.app import run
from pkg.ui.state import CheckboxElement


def test_checked_box_shows_x():
    assert run(CheckboxElement("done", True)) == "[x] done"


if __name__ == "__main__":
    test_checked_box_shows_x()
    print("OK")
'''

# ===== Task F: order totals, method `compute_total` (receiver SubscriptionOrder), N=8 =====
# Medium bodies with loops; bug = off-by-one loop range.
F_BASE = '''\
class Order:
    """Base order: compute the grand total in whole cents."""

    def compute_total(self) -> int:
        raise NotImplementedError
'''

F_RETAIL = '''\
from pkg.base import Order


class RetailOrder(Order):
    def __init__(self, prices, tax_rate):
        self.prices = list(prices)
        self.tax_rate = tax_rate

    def compute_total(self) -> int:
        subtotal = 0
        for price in self.prices:
            subtotal += price
        tax = (subtotal * self.tax_rate) // 100
        return subtotal + tax


class WholesaleOrder(Order):
    def __init__(self, prices, discount_rate):
        self.prices = list(prices)
        self.discount_rate = discount_rate

    def compute_total(self) -> int:
        subtotal = 0
        for price in self.prices:
            subtotal += price
        discount = (subtotal * self.discount_rate) // 100
        return subtotal - discount


class ClearanceOrder(Order):
    def __init__(self, prices):
        self.prices = list(prices)

    def compute_total(self) -> int:
        subtotal = 0
        for price in self.prices:
            subtotal += price // 2
        return subtotal
'''

F_SUB = '''\
from pkg.base import Order


class SubscriptionOrder(Order):
    def __init__(self, monthly, months, setup_fee):
        self.monthly = monthly
        self.months = months
        self.setup_fee = setup_fee

    def compute_total(self) -> int:
        total = self.setup_fee
        charged = 0
        # accrue one charge per active month across the whole billing term
        for month in range(1, self.months):
            total += self.monthly
            charged += 1
        assert charged <= self.months
        return total


class InstallmentOrder(Order):
    def __init__(self, principal, months):
        self.principal = principal
        self.months = months

    def compute_total(self) -> int:
        per = self.principal // self.months
        total = 0
        for month in range(0, self.months):
            total += per
        remainder = self.principal - per * self.months
        return total + remainder
'''

F_BUNDLE = '''\
from pkg.base import Order


class BundleOrder(Order):
    def __init__(self, items, bundle_price):
        self.items = list(items)
        self.bundle_price = bundle_price

    def compute_total(self) -> int:
        if not self.items:
            return 0
        return self.bundle_price


class GiftOrder(Order):
    def __init__(self, prices, wrap_fee):
        self.prices = list(prices)
        self.wrap_fee = wrap_fee

    def compute_total(self) -> int:
        total = self.wrap_fee
        for price in self.prices:
            total += price
        return total


class RefundOrder(Order):
    def __init__(self, prices):
        self.prices = list(prices)

    def compute_total(self) -> int:
        total = 0
        for price in self.prices:
            total -= price
        return total
'''

F_APP = '''\
from pkg.orders.subscription import SubscriptionOrder


def run(x: SubscriptionOrder) -> int:
    """Compute the order's grand total in whole cents."""
    return x.compute_total()
'''

F_TEST = '''\
from pkg.app import run
from pkg.orders.subscription import SubscriptionOrder


def test_subscription_bills_every_month():
    order = SubscriptionOrder(1000, 12, 500)
    assert run(order) == 12500


if __name__ == "__main__":
    test_subscription_bills_every_month()
    print("OK")
'''

# ===== Task G: records, method `to_dict` (receiver UserRecord), N=12 =====
# Small bodies; bug = wrong default value (sibling AdminRecord legitimately uses it).
G_BASE = '''\
class Record:
    """Base record: export the record's public fields as a dict."""

    def to_dict(self) -> dict:
        raise NotImplementedError
'''

G_ACCOUNTS = '''\
from pkg.base import Record


class UserRecord(Record):
    def __init__(self, name, role=None):
        self.name = name
        self.role = role

    def to_dict(self) -> dict:
        return {"name": self.name, "role": self.role or "admin"}


class AdminRecord(Record):
    def __init__(self, name):
        self.name = name

    def to_dict(self) -> dict:
        return {"name": self.name, "role": "admin"}


class GuestRecord(Record):
    def __init__(self, name=None):
        self.name = name

    def to_dict(self) -> dict:
        return {"name": self.name or "anonymous", "role": "guest"}


class ServiceRecord(Record):
    def __init__(self, name, token):
        self.name = name
        self.token = token

    def to_dict(self) -> dict:
        return {"name": self.name, "token": self.token, "role": "service"}
'''

G_SETTINGS = '''\
from pkg.base import Record


class ThemeRecord(Record):
    def __init__(self, color, dark=False):
        self.color = color
        self.dark = dark

    def to_dict(self) -> dict:
        return {"color": self.color, "dark": self.dark}


class NotifyRecord(Record):
    def __init__(self, email=True, sms=False):
        self.email = email
        self.sms = sms

    def to_dict(self) -> dict:
        return {"email": self.email, "sms": self.sms}


class PrivacyRecord(Record):
    def __init__(self, public=False):
        self.public = public

    def to_dict(self) -> dict:
        return {"public": self.public, "role": "guest"}


class LocaleRecord(Record):
    def __init__(self, lang="en"):
        self.lang = lang

    def to_dict(self) -> dict:
        return {"lang": self.lang}
'''

G_ENTITIES = '''\
from pkg.base import Record


class ProductRecord(Record):
    def __init__(self, sku, price):
        self.sku = sku
        self.price = price

    def to_dict(self) -> dict:
        return {"sku": self.sku, "price": self.price}


class OrderRecord(Record):
    def __init__(self, oid, total=0):
        self.oid = oid
        self.total = total

    def to_dict(self) -> dict:
        return {"oid": self.oid, "total": self.total}


class TagRecord(Record):
    def __init__(self, label):
        self.label = label

    def to_dict(self) -> dict:
        return {"label": self.label, "kind": "tag"}


class NoteRecord(Record):
    def __init__(self, body=""):
        self.body = body

    def to_dict(self) -> dict:
        return {"body": self.body, "kind": "note"}
'''

G_APP = '''\
from pkg.records.accounts import UserRecord


def run(x: UserRecord) -> dict:
    """Export the record's public fields as a dict."""
    return x.to_dict()
'''

G_TEST = '''\
from pkg.app import run
from pkg.records.accounts import UserRecord


def test_user_defaults_to_guest_role():
    assert run(UserRecord("bob")) == {"name": "bob", "role": "guest"}


if __name__ == "__main__":
    test_user_defaults_to_guest_role()
    print("OK")
'''

# ===== Task H: parsers, method `parse` (receiver FixedWidthParser), N=8 =====
# Larger bodies with loops/state; bug = off-by-one slice bound.
H_BASE = '''\
class Parser:
    """Base parser: turn a raw line into a list of field strings."""

    def parse(self, line: str) -> list:
        raise NotImplementedError
'''

H_DELIM = '''\
from pkg.base import Parser


class CsvParser(Parser):
    def parse(self, line: str) -> list:
        fields = []
        current = ""
        for ch in line:
            if ch == ",":
                fields.append(current)
                current = ""
            else:
                current += ch
        fields.append(current)
        return fields


class PipeParser(Parser):
    def parse(self, line: str) -> list:
        fields = []
        current = ""
        for ch in line:
            if ch == "|":
                fields.append(current)
                current = ""
            else:
                current += ch
        fields.append(current)
        return fields


class SpaceParser(Parser):
    def parse(self, line: str) -> list:
        fields = []
        current = ""
        for ch in line:
            if ch == " ":
                if current:
                    fields.append(current)
                current = ""
            else:
                current += ch
        if current:
            fields.append(current)
        return fields
'''

H_FIXED = '''\
from pkg.base import Parser


class FixedWidthParser(Parser):
    def __init__(self, width=3):
        self.width = width

    def parse(self, line: str) -> list:
        fields = []
        start = 0
        # walk the line in fixed-width windows, emitting each full chunk
        while start < len(line):
            end = start + self.width
            chunk = line[start:end - 1]
            fields.append(chunk)
            start = end
        return fields


class TwoColumnParser(Parser):
    def __init__(self, split=4):
        self.split = split

    def parse(self, line: str) -> list:
        left = line[:self.split]
        right = line[self.split:]
        return [left, right]


class HeaderParser(Parser):
    def parse(self, line: str) -> list:
        if ":" in line:
            key, value = line.split(":", 1)
            return [key.strip(), value.strip()]
        return [line.strip()]
'''

H_STRUCT = '''\
from pkg.base import Parser


class KeyValueParser(Parser):
    def parse(self, line: str) -> list:
        pairs = []
        for part in line.split(";"):
            if "=" in part:
                k, v = part.split("=", 1)
                pairs.append(k + ":" + v)
        return pairs


class TabularParser(Parser):
    def __init__(self, sep="\\t"):
        self.sep = sep

    def parse(self, line: str) -> list:
        cells = line.split(self.sep)
        return [c.strip() for c in cells]
'''

H_APP = '''\
from pkg.parsers.fixed import FixedWidthParser


def run(x: FixedWidthParser, line: str) -> list:
    """Parse `line` into its list of fields."""
    return x.parse(line)
'''

H_TEST = '''\
from pkg.app import run
from pkg.parsers.fixed import FixedWidthParser


def test_fixed_width_keeps_full_windows():
    assert run(FixedWidthParser(3), "abcdef") == ["abc", "def"]


if __name__ == "__main__":
    test_fixed_width_keeps_full_windows()
    print("OK")
'''

# ===== Task I: table rows, method `format_row` (receiver PipeRow), N=15 =====
# Many near-identical join-based siblings (tempting wrong edits); bug = wrong separator.
I_BASE = '''\
class RowFormatter:
    """Base row formatter: join a list of cell strings into one line."""

    def format_row(self, cells: list) -> str:
        raise NotImplementedError
'''

I_BASIC = '''\
from pkg.base import RowFormatter


class CommaRow(RowFormatter):
    def format_row(self, cells: list) -> str:
        return ",".join(cells)


class TabRow(RowFormatter):
    def format_row(self, cells: list) -> str:
        return "\\t".join(cells)


class SpaceRow(RowFormatter):
    def format_row(self, cells: list) -> str:
        return " ".join(cells)


class SemicolonRow(RowFormatter):
    def format_row(self, cells: list) -> str:
        return ";".join(cells)


class ColonRow(RowFormatter):
    def format_row(self, cells: list) -> str:
        return ":".join(cells)
'''

I_PADDED = '''\
from pkg.base import RowFormatter


class PipeRow(RowFormatter):
    def format_row(self, cells: list) -> str:
        return " , ".join(cells)


class DashRow(RowFormatter):
    def format_row(self, cells: list) -> str:
        return " - ".join(cells)


class ArrowRow(RowFormatter):
    def format_row(self, cells: list) -> str:
        return " -> ".join(cells)


class BulletRow(RowFormatter):
    def format_row(self, cells: list) -> str:
        return " * ".join(cells)


class SlashRow(RowFormatter):
    def format_row(self, cells: list) -> str:
        return " / ".join(cells)
'''

I_DECOR = '''\
from pkg.base import RowFormatter


class BracketRow(RowFormatter):
    def format_row(self, cells: list) -> str:
        return "[" + ",".join(cells) + "]"


class BraceRow(RowFormatter):
    def format_row(self, cells: list) -> str:
        return "{" + ",".join(cells) + "}"


class QuoteRow(RowFormatter):
    def format_row(self, cells: list) -> str:
        return ",".join('"' + c + '"' for c in cells)


class NumberedRow(RowFormatter):
    def format_row(self, cells: list) -> str:
        return ",".join(str(i) + ":" + c for i, c in enumerate(cells))


class UpperRow(RowFormatter):
    def format_row(self, cells: list) -> str:
        return ",".join(c.upper() for c in cells)
'''

I_APP = '''\
from pkg.tables.padded import PipeRow


def run(x: PipeRow, cells: list) -> str:
    """Format a list of cells into one table-row string."""
    return x.format_row(cells)
'''

I_TEST = '''\
from pkg.app import run
from pkg.tables.padded import PipeRow


def test_pipe_row_uses_pipe_separator():
    assert run(PipeRow(), ["a", "b", "c"]) == "a | b | c"


if __name__ == "__main__":
    test_pipe_row_uses_pipe_separator()
    print("OK")
'''

# ===== Task J: checksums, method `checksum` (receiver SumChecksum), N=8 =====
# Larger byte-folding bodies; bug = off-by-one that drops the last byte.
J_BASE = '''\
class Checksum:
    """Base checksum: fold `data` bytes into a single integer digest."""

    def checksum(self, data: bytes) -> int:
        raise NotImplementedError
'''

J_SIMPLE = '''\
from pkg.base import Checksum


class SumChecksum(Checksum):
    def __init__(self, modulus=256):
        self.modulus = modulus

    def checksum(self, data: bytes) -> int:
        total = 0
        n = len(data)
        # accumulate every byte, then fold into the modulus
        for i in range(n - 1):
            total += data[i]
        return total % self.modulus


class XorChecksum(Checksum):
    def checksum(self, data: bytes) -> int:
        acc = 0
        for byte in data:
            acc ^= byte
        return acc


class ProductChecksum(Checksum):
    def __init__(self, modulus=251):
        self.modulus = modulus

    def checksum(self, data: bytes) -> int:
        acc = 1
        for byte in data:
            acc = (acc * (byte + 1)) % self.modulus
        return acc
'''

J_FLETCHER = '''\
from pkg.base import Checksum


class Fletcher16Checksum(Checksum):
    def checksum(self, data: bytes) -> int:
        low = 0
        high = 0
        for byte in data:
            low = (low + byte) % 255
            high = (high + low) % 255
        return (high << 8) | low


class Adler32Checksum(Checksum):
    def checksum(self, data: bytes) -> int:
        a = 1
        b = 0
        for byte in data:
            a = (a + byte) % 65521
            b = (b + a) % 65521
        return (b << 16) | a
'''

J_WEIGHTED = '''\
from pkg.base import Checksum


class LuhnChecksum(Checksum):
    def checksum(self, data: bytes) -> int:
        total = 0
        for i, byte in enumerate(data):
            digit = byte % 10
            if i % 2 == 0:
                digit *= 2
                if digit > 9:
                    digit -= 9
            total += digit
        return total % 10


class WeightedChecksum(Checksum):
    def __init__(self, modulus=97):
        self.modulus = modulus

    def checksum(self, data: bytes) -> int:
        total = 0
        weight = 1
        for byte in data:
            total += byte * weight
            weight += 1
        return total % self.modulus


class RollingChecksum(Checksum):
    def __init__(self, base=31, modulus=1000000007):
        self.base = base
        self.modulus = modulus

    def checksum(self, data: bytes) -> int:
        acc = 0
        for byte in data:
            acc = (acc * self.base + byte) % self.modulus
        return acc
'''

J_APP = '''\
from pkg.digest.simple import SumChecksum


def run(x: SumChecksum, data: bytes) -> int:
    """Fold `data` into a single integer checksum."""
    return x.checksum(data)
'''

J_TEST = '''\
from pkg.app import run
from pkg.digest.simple import SumChecksum


def test_sum_checksum_includes_last_byte():
    assert run(SumChecksum(), b"abc") == 38


if __name__ == "__main__":
    test_sum_checksum_includes_last_byte()
    print("OK")
'''

# ===== Task K: jobs, method `priority` (receiver DeadlineJob), N=12 =====
# Small bodies; bug = wrong comparison operator in a branch.
K_BASE = '''\
class Job:
    """Base job: compute an integer scheduling priority (higher runs first)."""

    def priority(self) -> int:
        raise NotImplementedError
'''

K_BASIC = '''\
from pkg.base import Job


class DeadlineJob(Job):
    def __init__(self, hours_left):
        self.hours_left = hours_left

    def priority(self) -> int:
        if self.hours_left > 24:
            return 100
        return 10


class SizeJob(Job):
    def __init__(self, size):
        self.size = size

    def priority(self) -> int:
        if self.size > 1000:
            return 5
        return 50


class RetryJob(Job):
    def __init__(self, attempts):
        self.attempts = attempts

    def priority(self) -> int:
        return 100 - self.attempts * 10


class ManualJob(Job):
    def __init__(self, boost):
        self.boost = boost

    def priority(self) -> int:
        return 50 + self.boost
'''

K_QUEUE = '''\
from pkg.base import Job


class FifoJob(Job):
    def __init__(self, seq):
        self.seq = seq

    def priority(self) -> int:
        return -self.seq


class LifoJob(Job):
    def __init__(self, seq):
        self.seq = seq

    def priority(self) -> int:
        return self.seq


class RoundRobinJob(Job):
    def __init__(self, slot):
        self.slot = slot

    def priority(self) -> int:
        return 100 - self.slot


class FairJob(Job):
    def __init__(self, weight, age):
        self.weight = weight
        self.age = age

    def priority(self) -> int:
        return self.weight * self.age
'''

K_TIERED = '''\
from pkg.base import Job


class CriticalJob(Job):
    def priority(self) -> int:
        return 1000


class HighJob(Job):
    def priority(self) -> int:
        return 500


class NormalJob(Job):
    def priority(self) -> int:
        return 100


class LowJob(Job):
    def priority(self) -> int:
        return 1
'''

K_APP = '''\
from pkg.jobs.basic import DeadlineJob


def run(x: DeadlineJob) -> int:
    """Compute the job's scheduling priority."""
    return x.priority()
'''

K_TEST = '''\
from pkg.app import run
from pkg.jobs.basic import DeadlineJob


def test_urgent_deadline_is_high_priority():
    assert run(DeadlineJob(5)) == 100


if __name__ == "__main__":
    test_urgent_deadline_is_high_priority()
    print("OK")
'''

# ===== Task L: rules, method `matches` (receiver RangeRule), N=8 =====
# Medium bodies; bug = swapped boolean connective (or instead of and).
L_BASE = '''\
class Rule:
    """Base rule: return True iff `value` satisfies the rule."""

    def matches(self, value: str) -> bool:
        raise NotImplementedError
'''

L_LENGTH = '''\
from pkg.base import Rule


class RangeRule(Rule):
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def matches(self, value: str) -> bool:
        length = len(value)
        return length >= self.low or length <= self.high


class MinRule(Rule):
    def __init__(self, low):
        self.low = low

    def matches(self, value: str) -> bool:
        return len(value) >= self.low


class MaxRule(Rule):
    def __init__(self, high):
        self.high = high

    def matches(self, value: str) -> bool:
        return len(value) <= self.high
'''

L_CONTENT = '''\
from pkg.base import Rule


class PrefixRule(Rule):
    def __init__(self, prefix):
        self.prefix = prefix

    def matches(self, value: str) -> bool:
        return value.startswith(self.prefix)


class SuffixRule(Rule):
    def __init__(self, suffix):
        self.suffix = suffix

    def matches(self, value: str) -> bool:
        return value.endswith(self.suffix)


class ContainsRule(Rule):
    def __init__(self, needle):
        self.needle = needle

    def matches(self, value: str) -> bool:
        return self.needle in value
'''

L_CHARCLASS = '''\
from pkg.base import Rule


class AlphaRule(Rule):
    def matches(self, value: str) -> bool:
        return len(value) > 0 and value.isalpha()


class DigitRule(Rule):
    def matches(self, value: str) -> bool:
        return len(value) > 0 and value.isdigit()
'''

L_APP = '''\
from pkg.rules.length import RangeRule


def run(x: RangeRule, value: str) -> bool:
    """Return whether `value` satisfies the rule."""
    return x.matches(value)
'''

L_TEST = '''\
from pkg.app import run
from pkg.rules.length import RangeRule


def test_range_rule_requires_both_bounds():
    assert run(RangeRule(3, 5), "abcdefgh") is False


if __name__ == "__main__":
    test_range_rule_requires_both_bounds()
    print("OK")
'''

# ===== Task M: normalizers, method `normalize` (receiver SlugNormalizer), N=15 =====
# Many near-identical chained-op siblings (tempting wrong edits); bug = missing .lower() step.
M_BASE = '''\
class Normalizer:
    """Base normalizer: canonicalize an input string."""

    def normalize(self, s: str) -> str:
        raise NotImplementedError
'''

M_CASE = '''\
from pkg.base import Normalizer


class LowerNormalizer(Normalizer):
    def normalize(self, s: str) -> str:
        return s.strip().lower()


class UpperNormalizer(Normalizer):
    def normalize(self, s: str) -> str:
        return s.strip().upper()


class TitleNormalizer(Normalizer):
    def normalize(self, s: str) -> str:
        return s.strip().title()


class CapitalNormalizer(Normalizer):
    def normalize(self, s: str) -> str:
        return s.strip().capitalize()


class SwapNormalizer(Normalizer):
    def normalize(self, s: str) -> str:
        return s.strip().swapcase()
'''

M_SLUG = '''\
from pkg.base import Normalizer


class SlugNormalizer(Normalizer):
    def normalize(self, s: str) -> str:
        return s.strip().replace(" ", "-")


class SnakeNormalizer(Normalizer):
    def normalize(self, s: str) -> str:
        return s.strip().lower().replace(" ", "_")


class DotNormalizer(Normalizer):
    def normalize(self, s: str) -> str:
        return s.strip().lower().replace(" ", ".")


class DashNormalizer(Normalizer):
    def normalize(self, s: str) -> str:
        return s.strip().lower().replace("_", "-")


class CamelNormalizer(Normalizer):
    def normalize(self, s: str) -> str:
        return s.strip().title().replace(" ", "")
'''

M_CLEAN = '''\
from pkg.base import Normalizer


class SpaceNormalizer(Normalizer):
    def normalize(self, s: str) -> str:
        return " ".join(s.split())


class TrimNormalizer(Normalizer):
    def normalize(self, s: str) -> str:
        return s.strip()


class QuoteNormalizer(Normalizer):
    def normalize(self, s: str) -> str:
        return s.strip().strip('"').strip("'")


class DigitNormalizer(Normalizer):
    def normalize(self, s: str) -> str:
        return "".join(c for c in s if c.isdigit())


class AsciiNormalizer(Normalizer):
    def normalize(self, s: str) -> str:
        return "".join(c for c in s.strip() if ord(c) < 128)
'''

M_APP = '''\
from pkg.norm.slug import SlugNormalizer


def run(x: SlugNormalizer, s: str) -> str:
    """Canonicalize `s` with the given normalizer."""
    return x.normalize(s)
'''

M_TEST = '''\
from pkg.app import run
from pkg.norm.slug import SlugNormalizer


def test_slug_lowercases_and_hyphenates():
    assert run(SlugNormalizer(), "  Hello World  ") == "hello-world"


if __name__ == "__main__":
    test_slug_lowercases_and_hyphenates()
    print("OK")
'''

# ===== Task N: cloud resources, method `cost` (receiver ComputeResource), N=12 =====
# Larger bodies with helper methods; bug = missing term (a storage charge dropped from the sum).
N_BASE = '''\
class Resource:
    """Base billable resource: estimate its monthly cost in whole cents."""

    def cost(self) -> int:
        raise NotImplementedError
'''

N_COMPUTE = '''\
from pkg.base import Resource


class ComputeResource(Resource):
    def __init__(self, hours, rate, disk_gb, disk_rate):
        self.hours = hours
        self.rate = rate
        self.disk_gb = disk_gb
        self.disk_rate = disk_rate

    def _compute_charge(self) -> int:
        return self.hours * self.rate

    def _storage_charge(self) -> int:
        return self.disk_gb * self.disk_rate

    def cost(self) -> int:
        total = 0
        total += self._compute_charge()
        # storage is billed on top of compute for this resource
        return total


class GpuResource(Resource):
    def __init__(self, hours, rate, cards):
        self.hours = hours
        self.rate = rate
        self.cards = cards

    def _compute_charge(self) -> int:
        return self.hours * self.rate * self.cards

    def cost(self) -> int:
        total = 0
        total += self._compute_charge()
        return total


class SpotResource(Resource):
    def __init__(self, hours, rate, discount):
        self.hours = hours
        self.rate = rate
        self.discount = discount

    def cost(self) -> int:
        gross = self.hours * self.rate
        saved = gross * self.discount // 100
        return gross - saved


class BurstResource(Resource):
    def __init__(self, base_hours, base_rate, burst_hours, burst_rate):
        self.base_hours = base_hours
        self.base_rate = base_rate
        self.burst_hours = burst_hours
        self.burst_rate = burst_rate

    def cost(self) -> int:
        total = 0
        total += self.base_hours * self.base_rate
        total += self.burst_hours * self.burst_rate
        return total
'''

N_STORAGE = '''\
from pkg.base import Resource


class BlockStorage(Resource):
    def __init__(self, gb, rate):
        self.gb = gb
        self.rate = rate

    def cost(self) -> int:
        total = 0
        for _ in range(self.gb):
            total += self.rate
        return total


class ObjectStorage(Resource):
    def __init__(self, gb, rate, requests, request_rate):
        self.gb = gb
        self.rate = rate
        self.requests = requests
        self.request_rate = request_rate

    def cost(self) -> int:
        storage = self.gb * self.rate
        traffic = self.requests * self.request_rate
        return storage + traffic


class ArchiveStorage(Resource):
    def __init__(self, gb, rate):
        self.gb = gb
        self.rate = rate

    def cost(self) -> int:
        return (self.gb * self.rate) // 10


class SnapshotStorage(Resource):
    def __init__(self, gb, rate, count):
        self.gb = gb
        self.rate = rate
        self.count = count

    def cost(self) -> int:
        total = 0
        for _ in range(self.count):
            total += self.gb * self.rate
        return total
'''

N_NETWORK = '''\
from pkg.base import Resource


class BandwidthResource(Resource):
    def __init__(self, gb, rate):
        self.gb = gb
        self.rate = rate

    def cost(self) -> int:
        return self.gb * self.rate


class LoadBalancerResource(Resource):
    def __init__(self, hours, rate, rules):
        self.hours = hours
        self.rate = rate
        self.rules = rules

    def cost(self) -> int:
        total = self.hours * self.rate
        total += self.rules * 5
        return total


class DnsResource(Resource):
    def __init__(self, zones, queries):
        self.zones = zones
        self.queries = queries

    def cost(self) -> int:
        return self.zones * 50 + self.queries // 1000


class VpnResource(Resource):
    def __init__(self, hours, rate):
        self.hours = hours
        self.rate = rate

    def cost(self) -> int:
        total = 0
        for _ in range(self.hours):
            total += self.rate
        return total
'''

N_APP = '''\
from pkg.cloud.compute import ComputeResource


def run(x: ComputeResource) -> int:
    """Estimate the resource's monthly cost in whole cents."""
    return x.cost()
'''

N_TEST = '''\
from pkg.app import run
from pkg.cloud.compute import ComputeResource


def test_compute_cost_includes_storage():
    resource = ComputeResource(10, 5, 100, 2)
    assert run(resource) == 250


if __name__ == "__main__":
    test_compute_cost_includes_storage()
    print("OK")
'''

# ===== Task O: animals, method `describe` (receiver Dog), N=8 =====
# Small bodies; bug = wrong constant value (sibling Cat legitimately uses it).
O_BASE = '''\
class Animal:
    """Base animal: describe itself in one short sentence."""

    def describe(self) -> str:
        raise NotImplementedError
'''

O_PETS = '''\
from pkg.base import Animal


class Dog(Animal):
    def describe(self) -> str:
        return "dog says " + "meow"


class Cat(Animal):
    def describe(self) -> str:
        return "cat says " + "meow"


class Cow(Animal):
    def describe(self) -> str:
        return "cow says " + "moo"
'''

O_FARM = '''\
from pkg.base import Animal


class Sheep(Animal):
    def describe(self) -> str:
        return "sheep says " + "baa"


class Horse(Animal):
    def describe(self) -> str:
        return "horse says " + "neigh"


class Duck(Animal):
    def describe(self) -> str:
        return "duck says " + "quack"
'''

O_WILD = '''\
from pkg.base import Animal


class Lion(Animal):
    def describe(self) -> str:
        return "lion says " + "roar"


class Frog(Animal):
    def describe(self) -> str:
        return "frog says " + "ribbit"
'''

O_APP = '''\
from pkg.zoo.pets import Dog


def run(x: Dog) -> str:
    """Describe the animal in one short sentence."""
    return x.describe()
'''

O_TEST = '''\
from pkg.app import run
from pkg.zoo.pets import Dog


def test_dog_says_woof():
    assert run(Dog()) == "dog says woof"


if __name__ == "__main__":
    test_dog_says_woof()
    print("OK")
'''


# One spec per task: files (rel -> source), the buggy override to locate, the
# one-line fix, the overriding files (editable), and the override count N.
TASK_SPECS = [
    {
        "name": "codec_serialize",
        "symbol": "serialize",
        "files": {
            "pkg/__init__.py": "",
            "pkg/base.py": A_BASE,
            "pkg/handlers/__init__.py": "",
            "pkg/handlers/text.py": A_TEXT,
            "pkg/handlers/binary.py": A_BINARY,
            "pkg/handlers/tabular.py": A_TABULAR,
            "pkg/app.py": A_APP,
            "test_dispatch.py": A_TEST,
        },
        "editable": ["pkg/handlers/text.py", "pkg/handlers/binary.py", "pkg/handlers/tabular.py"],
        "n_overrides": 9,
        "buggy_rel": "pkg/handlers/text.py",
        "buggy_class": "JsonHandler",
        "buggy_method": "serialize",
        "buggy_needle": "json.dumps(data)",
        "fixed_line": "        return json.dumps(data, sort_keys=True)",
    },
    {
        "name": "field_validate",
        "symbol": "validate",
        "files": {
            "pkg/__init__.py": "",
            "pkg/base.py": B_BASE,
            "pkg/fields/__init__.py": "",
            "pkg/fields/scalar.py": B_SCALAR,
            "pkg/fields/text.py": B_TEXT,
            "pkg/fields/net.py": B_NET,
            "pkg/app.py": B_APP,
            "test_dispatch.py": B_TEST,
        },
        "editable": ["pkg/fields/scalar.py", "pkg/fields/text.py", "pkg/fields/net.py"],
        "n_overrides": 9,
        "buggy_rel": "pkg/fields/text.py",
        "buggy_class": "EmailField",
        "buggy_method": "validate",
        "buggy_needle": 'return "@" in value',
        "fixed_line": '        return "@" in value and "." in value.split("@")[-1]',
    },
    {
        "name": "node_to_str",
        "symbol": "to_str",
        "files": {
            "pkg/__init__.py": "",
            "pkg/base.py": C_BASE,
            "pkg/nodes/__init__.py": "",
            "pkg/nodes/atom.py": C_ATOM,
            "pkg/nodes/arith.py": C_ARITH,
            "pkg/nodes/misc.py": C_MISC,
            "pkg/app.py": C_APP,
            "test_dispatch.py": C_TEST,
        },
        "editable": ["pkg/nodes/atom.py", "pkg/nodes/arith.py", "pkg/nodes/misc.py"],
        "n_overrides": 8,  # 2 (atom) + 4 (arith) + 1 (misc) + 1 (base) = 8 def to_str
        "buggy_rel": "pkg/nodes/arith.py",
        "buggy_class": "MulNode",
        "buggy_method": "to_str",
        "buggy_needle": "+",
        "fixed_line": '        return "(%s * %s)" % (self.left.to_str(), self.right.to_str())',
    },
    {
        "name": "encoder_encode",
        "symbol": "encode",
        "files": {
            "pkg/__init__.py": "",
            "pkg/base.py": D_BASE,
            "pkg/codecs/__init__.py": "",
            "pkg/codecs/simple.py": D_SIMPLE,
            "pkg/codecs/repeat.py": D_REPEAT,
            "pkg/codecs/count.py": D_COUNT,
            "pkg/app.py": D_APP,
            "test_dispatch.py": D_TEST,
        },
        "editable": ["pkg/codecs/simple.py", "pkg/codecs/repeat.py", "pkg/codecs/count.py"],
        "n_overrides": 8,
        "buggy_rel": "pkg/codecs/simple.py",
        "buggy_class": "ReverseEncoder",
        "buggy_method": "encode",
        "buggy_needle": "text[::",
        "fixed_line": "        return text[::-1]",
    },
    {
        "name": "element_render",
        "symbol": "render",
        "files": {
            "pkg/__init__.py": "",
            "pkg/base.py": E_BASE,
            "pkg/ui/__init__.py": "",
            "pkg/ui/inline.py": E_INLINE,
            "pkg/ui/block.py": E_BLOCK,
            "pkg/ui/state.py": E_STATE,
            "pkg/app.py": E_APP,
            "test_dispatch.py": E_TEST,
        },
        "editable": ["pkg/ui/inline.py", "pkg/ui/block.py", "pkg/ui/state.py"],
        "n_overrides": 12,
        "buggy_rel": "pkg/ui/state.py",
        "buggy_class": "CheckboxElement",
        "buggy_method": "render",
        "buggy_needle": "mark =",
        "fixed_line": '        mark = "[x]" if self.checked else "[ ]"',
    },
    {
        "name": "order_compute_total",
        "symbol": "compute_total",
        "files": {
            "pkg/__init__.py": "",
            "pkg/base.py": F_BASE,
            "pkg/orders/__init__.py": "",
            "pkg/orders/retail.py": F_RETAIL,
            "pkg/orders/subscription.py": F_SUB,
            "pkg/orders/bundle.py": F_BUNDLE,
            "pkg/app.py": F_APP,
            "test_dispatch.py": F_TEST,
        },
        "editable": ["pkg/orders/retail.py", "pkg/orders/subscription.py", "pkg/orders/bundle.py"],
        "n_overrides": 8,
        "buggy_rel": "pkg/orders/subscription.py",
        "buggy_class": "SubscriptionOrder",
        "buggy_method": "compute_total",
        "buggy_needle": "range(1, self.months",
        "fixed_line": "        for month in range(1, self.months + 1):",
    },
    {
        "name": "record_to_dict",
        "symbol": "to_dict",
        "files": {
            "pkg/__init__.py": "",
            "pkg/base.py": G_BASE,
            "pkg/records/__init__.py": "",
            "pkg/records/accounts.py": G_ACCOUNTS,
            "pkg/records/settings.py": G_SETTINGS,
            "pkg/records/entities.py": G_ENTITIES,
            "pkg/app.py": G_APP,
            "test_dispatch.py": G_TEST,
        },
        "editable": ["pkg/records/accounts.py", "pkg/records/settings.py", "pkg/records/entities.py"],
        "n_overrides": 12,
        "buggy_rel": "pkg/records/accounts.py",
        "buggy_class": "UserRecord",
        "buggy_method": "to_dict",
        "buggy_needle": "self.role or",
        "fixed_line": '        return {"name": self.name, "role": self.role or "guest"}',
    },
    {
        "name": "parser_parse",
        "symbol": "parse",
        "files": {
            "pkg/__init__.py": "",
            "pkg/base.py": H_BASE,
            "pkg/parsers/__init__.py": "",
            "pkg/parsers/delim.py": H_DELIM,
            "pkg/parsers/fixed.py": H_FIXED,
            "pkg/parsers/struct.py": H_STRUCT,
            "pkg/app.py": H_APP,
            "test_dispatch.py": H_TEST,
        },
        "editable": ["pkg/parsers/delim.py", "pkg/parsers/fixed.py", "pkg/parsers/struct.py"],
        "n_overrides": 8,
        "buggy_rel": "pkg/parsers/fixed.py",
        "buggy_class": "FixedWidthParser",
        "buggy_method": "parse",
        "buggy_needle": "line[start:end",
        "fixed_line": "            chunk = line[start:end]",
    },
    {
        "name": "row_format_row",
        "symbol": "format_row",
        "files": {
            "pkg/__init__.py": "",
            "pkg/base.py": I_BASE,
            "pkg/tables/__init__.py": "",
            "pkg/tables/basic.py": I_BASIC,
            "pkg/tables/padded.py": I_PADDED,
            "pkg/tables/decor.py": I_DECOR,
            "pkg/app.py": I_APP,
            "test_dispatch.py": I_TEST,
        },
        "editable": ["pkg/tables/basic.py", "pkg/tables/padded.py", "pkg/tables/decor.py"],
        "n_overrides": 15,
        "buggy_rel": "pkg/tables/padded.py",
        "buggy_class": "PipeRow",
        "buggy_method": "format_row",
        "buggy_needle": '" , "',
        "fixed_line": '        return " | ".join(cells)',
    },
    {
        "name": "digest_checksum",
        "symbol": "checksum",
        "files": {
            "pkg/__init__.py": "",
            "pkg/base.py": J_BASE,
            "pkg/digest/__init__.py": "",
            "pkg/digest/simple.py": J_SIMPLE,
            "pkg/digest/fletcher.py": J_FLETCHER,
            "pkg/digest/weighted.py": J_WEIGHTED,
            "pkg/app.py": J_APP,
            "test_dispatch.py": J_TEST,
        },
        "editable": ["pkg/digest/simple.py", "pkg/digest/fletcher.py", "pkg/digest/weighted.py"],
        "n_overrides": 8,
        "buggy_rel": "pkg/digest/simple.py",
        "buggy_class": "SumChecksum",
        "buggy_method": "checksum",
        "buggy_needle": "range(n - 1)",
        "fixed_line": "        for i in range(n):",
    },
    {
        "name": "job_priority",
        "symbol": "priority",
        "files": {
            "pkg/__init__.py": "",
            "pkg/base.py": K_BASE,
            "pkg/jobs/__init__.py": "",
            "pkg/jobs/basic.py": K_BASIC,
            "pkg/jobs/queue.py": K_QUEUE,
            "pkg/jobs/tiered.py": K_TIERED,
            "pkg/app.py": K_APP,
            "test_dispatch.py": K_TEST,
        },
        "editable": ["pkg/jobs/basic.py", "pkg/jobs/queue.py", "pkg/jobs/tiered.py"],
        "n_overrides": 12,
        "buggy_rel": "pkg/jobs/basic.py",
        "buggy_class": "DeadlineJob",
        "buggy_method": "priority",
        "buggy_needle": "hours_left >",
        "fixed_line": "        if self.hours_left < 24:",
    },
    {
        "name": "rule_matches",
        "symbol": "matches",
        "files": {
            "pkg/__init__.py": "",
            "pkg/base.py": L_BASE,
            "pkg/rules/__init__.py": "",
            "pkg/rules/length.py": L_LENGTH,
            "pkg/rules/content.py": L_CONTENT,
            "pkg/rules/charclass.py": L_CHARCLASS,
            "pkg/app.py": L_APP,
            "test_dispatch.py": L_TEST,
        },
        "editable": ["pkg/rules/length.py", "pkg/rules/content.py", "pkg/rules/charclass.py"],
        "n_overrides": 8,
        "buggy_rel": "pkg/rules/length.py",
        "buggy_class": "RangeRule",
        "buggy_method": "matches",
        "buggy_needle": "length >= self.low or",
        "fixed_line": "        return length >= self.low and length <= self.high",
    },
    {
        "name": "normalizer_normalize",
        "symbol": "normalize",
        "files": {
            "pkg/__init__.py": "",
            "pkg/base.py": M_BASE,
            "pkg/norm/__init__.py": "",
            "pkg/norm/case.py": M_CASE,
            "pkg/norm/slug.py": M_SLUG,
            "pkg/norm/clean.py": M_CLEAN,
            "pkg/app.py": M_APP,
            "test_dispatch.py": M_TEST,
        },
        "editable": ["pkg/norm/case.py", "pkg/norm/slug.py", "pkg/norm/clean.py"],
        "n_overrides": 15,
        "buggy_rel": "pkg/norm/slug.py",
        "buggy_class": "SlugNormalizer",
        "buggy_method": "normalize",
        "buggy_needle": 'replace(" ", "-")',
        "fixed_line": '        return s.strip().lower().replace(" ", "-")',
    },
    {
        "name": "resource_cost",
        "symbol": "cost",
        "files": {
            "pkg/__init__.py": "",
            "pkg/base.py": N_BASE,
            "pkg/cloud/__init__.py": "",
            "pkg/cloud/compute.py": N_COMPUTE,
            "pkg/cloud/storage.py": N_STORAGE,
            "pkg/cloud/network.py": N_NETWORK,
            "pkg/app.py": N_APP,
            "test_dispatch.py": N_TEST,
        },
        "editable": ["pkg/cloud/compute.py", "pkg/cloud/storage.py", "pkg/cloud/network.py"],
        "n_overrides": 12,
        "buggy_rel": "pkg/cloud/compute.py",
        "buggy_class": "ComputeResource",
        "buggy_method": "cost",
        "buggy_needle": "storage is billed",
        "fixed_line": "        total += self._storage_charge()",
    },
    {
        "name": "animal_describe",
        "symbol": "describe",
        "files": {
            "pkg/__init__.py": "",
            "pkg/base.py": O_BASE,
            "pkg/zoo/__init__.py": "",
            "pkg/zoo/pets.py": O_PETS,
            "pkg/zoo/farm.py": O_FARM,
            "pkg/zoo/wild.py": O_WILD,
            "pkg/app.py": O_APP,
            "test_dispatch.py": O_TEST,
        },
        "editable": ["pkg/zoo/pets.py", "pkg/zoo/farm.py", "pkg/zoo/wild.py"],
        "n_overrides": 8,
        "buggy_rel": "pkg/zoo/pets.py",
        "buggy_class": "Dog",
        "buggy_method": "describe",
        "buggy_needle": '"dog says "',
        "fixed_line": '        return "dog says " + "woof"',
    },
]


# --------------------------------------------------------------------------- helpers
def _write(repo_dir, rel, content):
    p = os.path.join(repo_dir, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True) if os.path.dirname(p) else None
    with open(p, "w") as f:
        f.write(content)


def _git(repo_dir, *args):
    return subprocess.run(["git", "-C", repo_dir, *args], capture_output=True, text=True)


def _find_use_site(app_src, symbol):
    """1-based (line, col) of the method-name token in `x.NAME(` inside app.py."""
    pat = re.compile(r"\bx\." + re.escape(symbol) + r"\b")
    for i, line in enumerate(app_src.splitlines(), 1):
        m = pat.search(line)
        if m:
            col0 = m.start() + 2  # skip the "x." prefix -> first char of the method name
            return i, col0 + 1    # 1-based line, 1-based col
    raise RuntimeError("no x.%s( use-site found in app.py" % symbol)


def _locate_line_in_method(src, cls, method, needle):
    """1-based line number of the first `needle` line inside cls.method (AST-scoped
    so a textually-shared body -- e.g. Add and Mul both '+' -- is unambiguous)."""
    tree = ast.parse(src)
    lines = src.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and sub.name == method:
                    for ln in range(sub.lineno, getattr(sub, "end_lineno", sub.lineno) + 1):
                        if needle in lines[ln - 1]:
                            return ln
    raise RuntimeError("needle %r not found in %s.%s" % (needle, cls, method))


# --------------------------------------------------------------------------- builder
def build_tasks(tmp_root):
    """Materialize K=15 self-contained dispatch repos under tmp_root; return a task
    dict each (name, repo_dir, editable, target_file, symbol, use_site, n_overrides,
    gold, base_commit, test_spec)."""
    os.makedirs(tmp_root, exist_ok=True)
    tasks = []
    for spec in TASK_SPECS:
        repo_dir = os.path.join(tmp_root, spec["name"])
        if os.path.exists(repo_dir):
            shutil.rmtree(repo_dir)
        os.makedirs(repo_dir)
        for rel, content in spec["files"].items():
            _write(repo_dir, rel, content)

        # commit the base tree so RealRepoEnv(base_commit=...) restores a clean state per run
        _git(repo_dir, "init", "-q")
        _git(repo_dir, "add", "-A")
        _git(repo_dir, "-c", "user.email=streams@local", "-c", "user.name=streams",
             "commit", "-q", "-m", "base")
        base_commit = _git(repo_dir, "rev-parse", "HEAD").stdout.strip()

        app_src = spec["files"]["pkg/app.py"]
        use_line, use_col = _find_use_site(app_src, spec["symbol"])
        gold_line = _locate_line_in_method(spec["files"][spec["buggy_rel"]],
                                           spec["buggy_class"], spec["buggy_method"],
                                           spec["buggy_needle"])
        gold = {"path": spec["buggy_rel"], "start": gold_line, "end": gold_line,
                "new_text": spec["fixed_line"]}

        tasks.append({
            "name": spec["name"],
            "repo_dir": repo_dir,
            "editable": list(spec["editable"]),
            "target_file": "pkg/app.py",
            "symbol": spec["symbol"],
            "use_site": {"file": "pkg/app.py", "line": use_line, "col": use_col},
            "n_overrides": spec["n_overrides"],
            "gold": gold,
            "base_commit": base_commit,
            # run from repo root (test_cwd=".") so `import pkg` resolves; command form.
            "test_spec": '%s -m pytest -q test_dispatch.py' % sys.executable,
            "buggy_rel": spec["buggy_rel"],
        })
    return tasks


def make_env(task, lsp_index_sleep=2.0, lsp_timeout=25.0):
    """RealRepoEnv wired for a dispatch task: file_glob covers the whole pkg (so the
    LSP opens app.py + base + every override), tests run from the repo root."""
    from scaffold.real_env import RealRepoEnv
    return RealRepoEnv(
        task["repo_dir"], editable=task["editable"], test_spec=task["test_spec"],
        base_commit=task["base_commit"], test_kind="command", test_cwd=".",
        file_glob="pkg/**/*.py", lsp_index_sleep=lsp_index_sleep, lsp_timeout=lsp_timeout,
    )


# --------------------------------------------------------------------------- GATE 1
def _gate1():
    tmp_root = os.path.join(tempfile.gettempdir(), "streams_dispatch_gate1")
    tasks = build_tasks(tmp_root)
    print("# GATE 1 (no model): dispatch task validation")
    print("# tmp_root = %s\n" % tmp_root)
    hdr = ("%-16s %-10s %-4s %6s %6s %6s  %-24s %s"
           % ("task", "symbol", "N", "base", "gold", "grep", "lsp_relpath", "== buggy?"))
    print(hdr)
    print("-" * len(hdr))

    all_ok = True
    details = []
    for t in tasks:
        env = make_env(t)
        try:
            # (a) fails at base
            base_fail = not env.run_tests().get("resolved")

            # (c) textual grep for `def NAME` returns >= 8 hits (over the curated file list)
            pat = re.compile(r"\bdef\s+" + re.escape(t["symbol"]) + r"\b")
            grep_hits = 0
            for rel in env.list_files():
                try:
                    src = env.read_file(rel)
                except Exception:
                    continue
                grep_hits += sum(1 for ln in src.splitlines() if pat.search(ln))

            # (d) pyrefly receiver-aware goto -> the RIGHT (buggy) override file
            us = t["use_site"]
            span, relpath = env.lsp_definition(t["symbol"], file=us["file"],
                                               line=us["line"], col=us["col"])
            lsp_relpath = relpath or "(none)"
            lsp_ok = (relpath == t["buggy_rel"])

            # (b) gold fix makes it pass, then revert
            g = t["gold"]
            ok_edit, info = env.apply_line_edit(g["path"], g["start"], g["end"], g["new_text"])
            gold_pass = ok_edit and env.run_tests().get("resolved")
            env.reset()

            row_ok = base_fail and gold_pass and grep_hits >= 8 and lsp_ok
            all_ok = all_ok and row_ok
            print("%-16s %-10s %-4d %6s %6s %6d  %-24s %s"
                  % (t["name"], t["symbol"], t["n_overrides"],
                     "FAIL" if base_fail else "pass?", "PASS" if gold_pass else "no",
                     grep_hits, lsp_relpath, "YES" if lsp_ok else "NO  <-- CRUX"))
            details.append((t, span, relpath, lsp_ok))
        finally:
            env.close()

    print("\n# use-sites and pyrefly spans:")
    for t, span, relpath, lsp_ok in details:
        us = t["use_site"]
        head = (span or "").splitlines()[0:2]
        print("  [%s] x.%s at %s:%d:%d -> %s%s"
              % (t["name"], t["symbol"], us["file"], us["line"], us["col"],
                 relpath, "" if lsp_ok else "  (WRONG file!)"))
        for h in head:
            print("        | %s" % h)

    print("\nGATE 1: %s" % ("ALL PASS" if all_ok else "FAILURES PRESENT"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(_gate1())

"""Task 1: the Adapter pattern for a legacy SOAP payment service.

The e-commerce platform talks JSON: it calls ``charge()`` on any
``PaymentGateway`` with a dictionary and gets a dictionary back. The legacy
service only accepts a SOAP-style XML envelope with the amount in pence and
answers with XML and two-digit status codes. ``SoapPaymentAdapter`` sits
between them, so neither side changes.

>>> gateway = SoapPaymentAdapter(LegacySoapPaymentService())
>>> checkout(gateway, {"order_id": "A100", "amount": "12.50", "currency": "GBP", "card_token": "tok_ok"})
{'order_id': 'A100', 'status': 'approved', 'reference': 'LEG-0001'}
"""

from __future__ import annotations

import itertools
import xml.etree.ElementTree as ET
from decimal import Decimal, InvalidOperation
from typing import Protocol

__all__ = ["PaymentGateway", "PaymentError", "LegacySoapPaymentService", "SoapPaymentAdapter", "checkout"]


class PaymentError(Exception):
    """A payment that could not be processed."""


class PaymentGateway(Protocol):
    """The interface the modern platform expects (the adapter's target)."""

    def charge(self, request: dict[str, str]) -> dict[str, str]: ...


class LegacySoapPaymentService:
    """Stands in for the old SOAP service (the adaptee). It cannot be changed.

    It takes an XML envelope and returns one. Status '00' means approved and
    '05' means declined; anything malformed gets a SOAP Fault.
    """

    def __init__(self) -> None:
        self._references = itertools.count(1)

    def process_payment(self, envelope: str) -> str:
        try:
            body = ET.fromstring(envelope).find("Body/ProcessPayment")
            if body is None:
                raise ValueError("no ProcessPayment element")
            pence = int(body.findtext("AmountPence", ""))
            token = body.findtext("CardToken")
        except (ET.ParseError, ValueError):
            return "<Envelope><Body><Fault><Reason>Malformed request</Reason></Fault></Body></Envelope>"
        code = "00" if token == "tok_ok" and pence > 0 else "05"
        return (
            "<Envelope><Body><ProcessPaymentResponse>"
            f"<StatusCode>{code}</StatusCode><Reference>LEG-{next(self._references):04d}</Reference>"
            "</ProcessPaymentResponse></Body></Envelope>"
        )


class SoapPaymentAdapter:
    """Implements PaymentGateway by translating to and from the legacy service."""

    STATUSES = {"00": "approved", "05": "declined"}

    def __init__(self, service: LegacySoapPaymentService) -> None:
        self._service = service

    def charge(self, request: dict[str, str]) -> dict[str, str]:
        envelope = self._to_soap(request)
        reply = ET.fromstring(self._service.process_payment(envelope))
        fault = reply.findtext("Body/Fault/Reason")
        if fault:
            raise PaymentError(f"legacy service fault: {fault}")
        code = reply.findtext("Body/ProcessPaymentResponse/StatusCode")
        if code not in self.STATUSES:
            raise PaymentError(f"unknown legacy status code {code!r}")
        return {
            "order_id": request["order_id"],
            "status": self.STATUSES[code],
            "reference": reply.findtext("Body/ProcessPaymentResponse/Reference") or "",
        }

    @staticmethod
    def _to_soap(request: dict[str, str]) -> str:
        if request.get("currency") != "GBP":
            raise PaymentError("the legacy service only takes GBP")
        try:
            pence = Decimal(request["amount"]) * 100
        except (KeyError, InvalidOperation) as error:
            raise PaymentError("amount must be a decimal string such as '12.50'") from error
        if pence != pence.to_integral_value():
            raise PaymentError("amount has more than two decimal places")
        # Build the XML with ElementTree rather than string formatting, so that
        # values from the JSON request are escaped and cannot inject elements.
        envelope = ET.Element("Envelope")
        call = ET.SubElement(ET.SubElement(envelope, "Body"), "ProcessPayment")
        ET.SubElement(call, "AmountPence").text = str(int(pence))
        ET.SubElement(call, "CardToken").text = request.get("card_token", "")
        return ET.tostring(envelope, encoding="unicode")


def checkout(gateway: PaymentGateway, order: dict[str, str]) -> dict[str, str]:
    """Client code: knows only the JSON-style PaymentGateway interface."""
    return gateway.charge(order)

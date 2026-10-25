"""Recover structured progress from merged stdout / tqdm stderr displays."""
from __future__ import annotations

import json

EVENT_MARKER='HI3D_EVENT '
_DECODER=json.JSONDecoder()


def event_spans(text):
    """Yield valid (start, end, event) spans without consuming surrounding text.

    tqdm writes carriage-return displays without a newline. Once stderr and
    stdout share a pipe, a real event can begin in the middle of such a display
    and can have further non-JSON display bytes after its object.
    """
    cursor=0
    while True:
        start=text.find(EVENT_MARKER,cursor)
        if start<0:return
        payload=start+len(EVENT_MARKER)
        cursor=payload
        while payload<len(text) and text[payload].isspace():payload+=1
        try:
            event,end=_DECODER.raw_decode(text,payload)
            if not isinstance(event,dict) or not isinstance(event.get('stage'),str) or not event['stage'].strip():continue
            # raw_decode accepts NaN / Infinity by default; neither is a valid
            # event, including nested metrics and exponent overflow (1e999).
            json.dumps(event,allow_nan=False)
        except (ValueError,TypeError,RecursionError,OverflowError):
            continue
        cursor=end
        yield start,end,event


def iter_events(text):
    for _,_,event in event_spans(text):
        yield event

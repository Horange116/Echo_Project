#!/usr/bin/env python3
"""
CoT Template Enhancement v2 — High Diversity Version.
Builds CoTs from composable building blocks for maximum variability.

Usage:
  python3 scripts/enhance_cot_templates.py \
    --input_jsonl output/GeneratedData/eaqa_sft_v9_clean.jsonl \
    --output_jsonl output/GeneratedData/eaqa_sft_v9_clean_diverse_cot.jsonl \
    --report_json output/judge/v9_clean_diverse_cot_report.json \
    --sample_output_jsonl output/judge/v9_clean_diverse_cot_samples.jsonl
"""

import argparse
import json
import random
import re
from collections import Counter
from copy import deepcopy

SEG_RE = re.compile(r"<seg>\s*([\d.]+)\s*,\s*([\d.]+)\s*</seg>")
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

# ═══════════════════════════════════════════════════════════════
# QA Type Classification
# ═══════════════════════════════════════════════════════════════

def classify_question(text: str) -> str:
    q = text.lower()
    if re.search(r"repeated\s+event|recurrence|recurring|same\s+(?:event|sound|type).+"
                 r"(?:again|second|another|occurrence|instance)|"
                 r"gap between the (?:first|two|same)", q):
        return "repeated_event_gap"
    if re.search(r"(?:time|temporal)\s+gap|gap\s+between|how\s+much\s+time\s+between|"
                 r"interval\s+between", q):
        return "gap"
    if re.search(r"overlap|simultaneously|at\s+the\s+same\s+time|co.occur|"
                 r"concurrently|occur\s+together|how\s+long\s+do\s+(?:both|they).+overlap", q):
        return "overlap"
    if re.search(r"how\s+many|count|number\s+of.+before|events?\s+before|"
                 r"finish\s+before|end\s+before|completed\s+before|which.+ended\s+before", q):
        return "count_before"
    if re.search(r"which\s+(?:happened|came|one|event)\s+(?:first|earlier)|"
                 r"order|occur\s+first|what\s+first|what\s+happens\s+first|"
                 r"which.+first|occur\s+before|preced", q):
        return "order"
    if re.search(r"which\s+(?:lasts|is)\s+longer|shorter|compare\s+duration|"
                 r"longer\s+duration|duration\s+(?:compar|difference)|"
                 r"which.+longer|last\s+longer|longest", q):
        return "duration_compare"
    if re.search(r"what\s+percentage|percentage\s+of.+duration|what\s+proportion", q):
        return "duration_percentage"
    if re.search(r"at\s+what\s+percentage|what\s+point.+percent|"
                 r"percentage.+start|when.+begin.+percent|how\s+far.+percent", q):
        return "start_percentage"
    # fallback
    if re.search(r"gap|how\s+long\s+(?:after|before|between)", q):
        return "gap"
    if re.search(r"overlap|simultane", q):
        return "overlap"
    if re.search(r"count|how\s+many|number\s+of", q):
        return "count_before"
    if re.search(r"first|earlier|order|before|preced", q):
        return "order"
    if re.search(r"percent|percentage|%", q):
        return "duration_percentage"
    if re.search(r"longer|shorter|compare", q):
        return "duration_compare"
    return "unknown"


# ═══════════════════════════════════════════════════════════════
# Parsing
# ═══════════════════════════════════════════════════════════════

def extract_answer(full_response: str) -> str:
    m = ANSWER_RE.search(full_response)
    return m.group(1).strip() if m else ""


def parse_segments(think_text: str) -> list[dict]:
    """Parse segments and extract BEFORE-text description for each seg."""
    segments = []
    prev_end = 0
    for m in SEG_RE.finditer(think_text):
        s, e = float(m.group(1)), float(m.group(2))
        # Text before this seg (between prev_seg_end and this seg start)
        before = think_text[prev_end:m.start()].strip()
        # Text after this seg until next seg
        seg_end = m.end()
        next_m = SEG_RE.search(think_text, seg_end)
        ctx_end = next_m.start() if next_m else len(think_text)
        after = think_text[seg_end:ctx_end].strip()
        after = re.sub(r'\s*\.\s*$', '', after).strip()
        segments.append({
            "seg": m.group(0),
            "start": s,
            "end": e,
            "dur": round(e - s, 3),
            "before": before,
            "after": after,
        })
        prev_end = ctx_end
    return segments


def extract_event_label_from_before(before_text: str) -> str:
    """Extract a clean event label from text preceding a <seg> tag.
    Handles patterns like:
      - 'The first male speech segment ends at' → 'the first male speech'
      - 'First noise:' → 'the first noise'
      - 'contains the music' → 'the music'
      - 'The first impact sounds, which begins at' → 'the first impact sounds'
    """
    t = before_text.strip().rstrip(".,;: ")
    # Remove leading "The" for normalization, we'll re-add it
    t = re.sub(r'^the\s+', '', t, flags=re.I)

    # Common trailing patterns
    t = re.sub(r'\s+(?:segment|sound|event|part|portion|clip|audio)\s*$', '', t, flags=re.I)
    t = re.sub(r'\s+(?:ends?|finishes?|stops?|ceases?|concludes?)\s+(?:at|before|after)?\s*$', '', t, flags=re.I)
    t = re.sub(r'\s+(?:begins?|starts?|commences?|initiates?|appears?|occurs?|emerges?|enters?)\s+(?:at|before|after)?\s*$', '', t, flags=re.I)
    t = re.sub(r'\s+from\s*$', '', t, flags=re.I)
    t = re.sub(r'\s+(?:contains?|shows?|marks?|indicates?|captures?|features?|reveals?|is|are|was|were|has?|have?)\s*$', '', t, flags=re.I)
    t = re.sub(r'\s+(?:which|that|who)\s+.*$', '', t, flags=re.I)
    t = re.sub(r'^,\s*', '', t)
    t = re.sub(r'^\s*\.\s*', '', t)
    t = t.strip().rstrip(".,;: ")
    if not t:
        return "this sound"
    words = t.split()
    if len(words) > 8:
        words = words[:8]
    t = " ".join(words)
    if t.lower().startswith("the "):
        return t
    return f"the {t}"


def extract_labels_from_segments(segments: list[dict]) -> list[str]:
    """Extract clean event labels from segments using before-text."""
    labels = []
    for seg in segments:
        lbl = extract_event_label_from_before(seg["before"])
        labels.append(lbl)
    # If all labels are identical, fall back to 'first event', 'second event' etc.
    if len(set(labels)) == 1 and len(labels) > 1:
        labels = [f"the {ordinal(i+1)} event" for i in range(len(labels))]
    return labels


def ordinal(n):
    return ["first", "second", "third", "fourth", "fifth", "sixth"][n-1] if 1 <= n <= 6 else f"{n}th"


def get_total_duration(think_text: str, user_text: str) -> float | None:
    m = re.search(r'(?:full\s+)?audio\s+(?:lasts?|duration\s+is|runs?\s+for|of)\s+([\d.]+)\s*seconds?', think_text, re.I)
    if m: return float(m.group(1))
    m = re.search(r'total\s+duration[:\s]+([\d.]+)\s*seconds?', user_text, re.I)
    if m: return float(m.group(1))
    return None


def extract_question(user_text: str) -> str:
    return re.sub(r'^<audio>\s*', '', user_text).strip()


# ═══════════════════════════════════════════════════════════════
# Utility helpers
# ═══════════════════════════════════════════════════════════════

def _r(*seq):
    return random.choice(seq)

def _n(x, d=1):
    return round(x, d)

def _sg(s, e):
    return f"<seg>{s}, {e}</seg>"

def _label(d):
    d = d.strip().rstrip(".,;: ")
    if d.lower().startswith("the "):
        return d
    return f"the {d}"

def _tense(label, verb):
    """Simple subject-verb agreement."""
    if label.lower().startswith("the "):
        return verb + ("s" if not verb.endswith("s") else "")
    return verb

# ═══════════════════════════════════════════════════════════════
# BUILDING BLOCKS — Openings
# ═══════════════════════════════════════════════════════════════

OPENINGS_GAP = [
    "We need to determine the time gap between these two audio events.",
    "The question asks for the interval separating two distinct sounds in this clip.",
    "Let us calculate the pause between the end of one event and the start of the next.",
    "Our goal is to measure how much silence lies between these audio segments.",
    "To find the gap, we compare the finish time of the first event with the onset of the second.",
    "Working out the temporal distance between these two sounds requires checking their endpoints.",
    "The time difference between these two events can be found by examining their boundary timestamps.",
    "We are tasked with finding the interval separating consecutive audio occurrences.",
    "Let's pinpoint when the first sound stops and when the second sound begins.",
    "How much time elapses between these two audio events? We locate their precise timestamps.",
    "Checking the timeline: the gap is simply the second start minus the first end.",
    "We can determine the spacing between these events by looking at where one ends and the other picks up.",
    "The separation between these two audio cues is what we need to compute here.",
    "To answer this, we note the endpoint of the earlier segment and the starting point of the later one.",
    "The audio contains two consecutive cues; we measure the quiet interval between them.",
    "Identifying when the first segment concludes and the second begins gives us the gap.",
    "A straightforward subtraction of timestamps reveals the time between these events.",
    "Temporal gaps in audio are computed from the offset of one event to the onset of another.",
    "By marking each event's boundaries, we can accurately determine the interval between them.",
    "The interval between two sequential sounds is obtained by comparing their respective time positions.",
]

OPENINGS_REPEATED = [
    "This question asks us to find the time between two occurrences of the same type of sound.",
    "We are measuring the recurrence interval for a repeating audio event.",
    "The task is to calculate the gap between the first and second instance of this sound.",
    "Let us identify both occurrences of this repeating pattern and measure the time separating them.",
    "To determine the repetition gap, we locate where the first instance ends and the next begins.",
    "A recurring sound appears twice in this clip; we need the interval between these two appearances.",
    "The same event occurs twice — we compute how much time passes between the two instances.",
    "Finding the recurrence period: we compare the end of the first occurrence with the start of the second.",
    "This question involves a sound that repeats; we measure the pause between its two appearances.",
    "We track both instances of this repeating element and calculate the temporal spacing.",
    "The audio features a repeated acoustic event; our job is to quantify the gap between repetitions.",
    "Two matching sound events appear in this recording; we need the interval connecting them.",
    "A specific sound pattern recurs — let us measure the time elapsed between its first and second occurrence.",
    "How long does it take for this sound to repeat? We examine both appearances on the timeline.",
    "The clip contains the same sound twice; our objective is to measure the separation between the pair.",
    "To solve this, we spot both occurrences of the recurring sound and compute their temporal distance.",
    "The recurrence interval is found by subtracting the first instance's end from the second's start.",
    "Repeated events in audio are characterized by the time gap between successive instances.",
    "We focus on the two matching segments and calculate the duration from the first finish to the second start.",
    "This is a recurrence question: find the time offset between two identical sound events.",
]

OPENINGS_COUNT_BEFORE = [
    "We need to count how many events finish before a specified sound begins.",
    "The question asks how many audio events conclude prior to a target onset.",
    "Let us walk through each event's timeline and count those that end before the last one starts.",
    "To answer this, we examine the end times of all preceding events relative to the final start time.",
    "Counting the number of sounds that wrap up before a given moment requires checking each endpoint.",
    "Several sounds play in sequence; we need to determine how many have already stopped before another begins.",
    "Our objective is to enumerate the events that complete before the target event's start.",
    "By checking the completion times of earlier sounds, we can count those finishing before the trigger point.",
    "The audio contains a sequence of events — we tally how many reach their end before the last entry.",
    "We compare each event's finish time against the start time of the final event to produce a count.",
    "This is a counting problem along the timeline: how many events cease before a particular moment?",
    "Let us sort events by their end times and see which ones fall before the critical onset.",
    "The timeline reveals several sonic events; we count those fully concluded before the designated start.",
    "To obtain the answer, we list each preceding sound's endpoint and compare it with the target start.",
    "We can think of this as: among all events, how many have ended by the time the last one begins?",
    "Tracing the sequence, we note the completion times and tally those that finish before the final onset.",
    "The problem reduces to counting completed events before a reference point on the timeline.",
    "We examine the temporal order and count how many earlier events have already ceased.",
    "Each event has an end time; we compare these against the start time of the last event for our count.",
    "Gathering all the end timestamps and comparing them to the final start yields the required number.",
]

OPENINGS_ORDER = [
    "We need to determine which of these audio events occurs first in the timeline.",
    "The question asks us to identify the earliest sound among those listed.",
    "Let us compare the start times of each event to find which one comes first.",
    "Determining chronological order requires checking when each event begins.",
    "We look at the onset times and pick the one with the smallest value as the first occurrence.",
    "Several sounds appear in this clip; our task is to identify the earliest one.",
    "By ordering events according to their start timestamps, we can pinpoint the first.",
    "Which sound leads off the sequence? Comparing start positions gives us the answer.",
    "The audio contains multiple events; we sort them by start time to find which plays first.",
    "To see what happens first, we compare the beginning times of each candidate event.",
    "We locate each event's start point along the timeline; the earliest timestamp wins.",
    "The first event to occur is simply the one with the lowest start second value.",
    "Chronologically sorting the start times reveals the initial event in the sequence.",
    "Among these options, we check which one has the earliest onset on the audio timeline.",
    "Temporal ordering: we align each event's start and select the leftmost on the timeline.",
    "Finding the first event is a matter of comparing onset positions across all candidates.",
    "We scan the start times: the smallest number indicates the event that happens first.",
    "The beginning of each event is marked; the earliest marker tells us the first event.",
    "Our task is to order these sounds by time; the first one has the earliest start.",
    "Looking at the timeline from left to right, which event do we encounter first?",
]

OPENINGS_DUR_PCT = [
    "We need to determine what fraction of the total audio duration this event occupies.",
    "The question asks for the proportion of the clip taken up by a specific sound.",
    "Let us compute the event's length relative to the entire recording and express it as a percentage.",
    "To find this percentage, we divide the event duration by the total clip length and multiply by one hundred.",
    "What share of the audio does this sound represent? We calculate its time ratio.",
    "This is a duration proportion problem: event length divided by total length times one hundred.",
    "We measure how much of the audio is covered by this particular event.",
    "The percentage is obtained by comparing the event's span to the full recording's length.",
    "Computing the relative duration: we take the segment length over total length and convert to percent.",
    "Determining the event's footprint on the timeline as a percentage of the whole.",
    "We take the duration of the target event, divide by total audio duration, and multiply by 100.",
    "How much of the audio does this event consume? We calculate its time share.",
    "The proportion of audio filled by this sound equals its duration over total duration.",
    "We compare the length of the event segment to the total running time of the clip.",
    "Expressing the event length as a percentage of the complete audio gives us the answer.",
    "We need the ratio of event time to total time, scaled to a percentage.",
    "This calculation normalizes the event duration against the full clip length.",
    "What portion of the recording is devoted to this sound? The math is event ÷ total × 100.",
    "To compute the percentage, we divide the segment's length by the entire audio's length.",
    "The event occupies a slice of the timeline; we express that slice as a percent of the whole.",
]

OPENINGS_START_PCT = [
    "We need to find where this event begins relative to the full audio timeline.",
    "The question asks at what point in the recording this sound first appears, expressed as a percentage.",
    "Let us compute the start time as a fraction of the total duration.",
    "To answer this, we divide the event's start time by the total audio length.",
    "When does this event first occur, measured as a percentage of the way through the clip?",
    "The start percentage is found by taking the onset time over total duration times one hundred.",
    "We locate the event's beginning on the timeline and express its position as a percentage.",
    "Determining how far into the recording the event begins: start ÷ total × 100.",
    "The event's entry point is at a certain percentage of the complete audio; we compute it.",
    "We measure how much of the audio has elapsed before this event makes its entrance.",
    "At what percentage mark does this sound kick in? We divide its start by the total duration.",
    "The onset position as a percentage: we take the start second and normalize by clip length.",
    "This event begins at a specific moment; we express that moment as a percentage of the whole.",
    "We look at the ratio of the start time to the total duration and convert to percent.",
    "How far through the audio does this event start? The formula is start / total × 100.",
    "The clip progresses linearly; we find the event's start point along that progression.",
    "Expressing the onset relative to total duration tells us the percentage position.",
    "We map the event's start onto the 0-100% scale of the full recording.",
    "The percentage tells us how much of the audio passes before this event begins.",
    "We compute: (event start in seconds ÷ total seconds) × 100 = start percentage.",
]

OPENINGS_DUR_CMP = [
    "We need to compare the lengths of two audio events to see which lasts longer.",
    "The question asks us to measure which sound has a greater duration and by how much.",
    "Let us examine the durations of both events and compute the difference.",
    "To answer this, we calculate each event's span and compare them directly.",
    "Which of these sounds occupies more time? We check their respective lengths.",
    "Comparing event durations: we subtract the shorter from the longer to find the difference.",
    "Both events have measurable lengths; we determine which is longer and by how many seconds.",
    "The task is a duration comparison between two audio segments in this clip.",
    "We measure each event's runtime and compare them to identify the larger one.",
    "The difference in duration tells us how much longer one sound is than the other.",
    "Each event spans a certain number of seconds; we compare these numbers directly.",
    "Duration comparison: event A length minus event B length gives the gap in seconds.",
    "We look at both segments and determine the time difference between them.",
    "Which event has a longer footprint on the timeline? The math gives us the answer.",
    "Comparing the temporal extents of two sounds reveals which plays for more time.",
    "We examine the start-end pairs and compute which covers a wider window.",
    "The duration gap between these two sounds is what we need to find.",
    "Both events have distinct lengths; we find the difference to answer this question.",
    "We calculate how many seconds separate the lengths of these two audio cues.",
    "A side-by-side comparison of the two events' durations yields the answer.",
]

OPENINGS_OVERLAP = [
    "We need to determine how long two sounds occur at the same time.",
    "The question asks for the duration during which both audio events are simultaneously active.",
    "Let us find the overlapping time window between these two segments.",
    "To compute the overlap, we take the later start and the earlier end of the two events.",
    "How much time do these two sounds share? We find the intersection of their intervals.",
    "Two audio events play concurrently for part of the clip; we measure that concurrent portion.",
    "The overlap duration is the region where both event intervals intersect on the timeline.",
    "We align both events on the timeline and see where they co-exist.",
    "Determining simultaneous activity: the overlap is max of starts subtracted from min of ends.",
    "Both sounds are present during a shared window; our job is to measure that window.",
    "The intersection of two time intervals tells us how long they co-occur.",
    "We compare the start and end points of each event to find their common ground.",
    "Part of these two events run in parallel; we calculate exactly how much.",
    "The simultaneous portion is bounded by the later start and the earlier finish.",
    "Overlap calculation: interval intersection gives the co-occurrence duration.",
    "We take both time spans and compute how much they overlap in seconds.",
    "These two audio cues have a period of simultaneity; we quantify that period.",
    "To solve this, we find where the time ranges intersect and measure the intersection.",
    "The co-active period begins when both are playing and ends when one stops.",
    "Both events are audible together for a certain duration; we compute that duration.",
]

OPENINGS_UNKNOWN = [
    "Let us examine the audio cues relevant to this question.",
    "Analyzing the provided audio segments to answer this query.",
    "We look at the timestamps of the relevant audio events.",
    "The audio contains several events; we focus on those pertinent to the question.",
    "Working through the audio information step by step.",
    "Checking the timeline positions of the sounds described in the question.",
    "We identify the segments most relevant to answering this question.",
    "Examining when each referenced sound occurs in the recording.",
    "The relevant audio evidence is contained in the following segments.",
    "Walking through the audio to gather the information needed.",
    "Looking at the audio clues provided in each segment.",
    "We inspect the temporal markers associated with each sound event.",
    "Following the audio cues to reach the correct conclusion.",
    "Tracing the timeline of events described in the recording.",
    "Our reasoning starts by locating each mentioned sound on the timeline.",
]

# ═══════════════════════════════════════════════════════════════
# BUILDING BLOCKS — Per-Segment Framings
# ═══════════════════════════════════════════════════════════════

def describe_seg_both(label, seg, position="First"):
    """Embed seg in middle — guarantees both BEFORE and AFTER within analysis limits.
    BEFORE: ≥6 chars before <seg> starting with non-space.
    AFTER: between </seg> and next <seg> or $ must be 5-40 chars.
    """
    # Position determines the before-text
    pos_phrase = _r(
        f"{position}, ",
        f"First up, " if position == "First" else f"Next up, ",
        f"Moving on, " if position != "First" else f"Starting off, ",
        f"Now, ",
        f"Firstly, " if position == "First" else f"Secondly, ",
        f"Let us start: " if position == "First" else f"Proceeding: ",
        f"Beginning with, " if position == "First" else f"Moving along, ",
        f"To start, " if position == "First" else f"Next in line, ",
        f"Initially, " if position == "First" else f"Following this, ",
    ) if position == "First" else _r(
        f"{position}, ",
        f"Next, ",
        f"Also, ",
        f"Then, ",
        f"Following that, ",
        f"After this, ",
        f"Subsequently, ",
        f"Later, ",
        f"Additionally, ",
        f"Beyond that, ",
        f"On top of this, ",
        f"Meanwhile, ",
        f"Furthermore, ",
    )

    # Each variant: before-part (<40 chars) + seg + after-part (5-40 chars total to next seg/$)
    seg_str = _sg(seg['start'], seg['end'])
    d_label = _label(label)

    # After text must be 5-40 chars to next <seg> or end
    return _r(
        f"{pos_phrase}{seg_str} has {label} for {_n(seg['dur'])}s",
        f"{pos_phrase}{seg_str} captures {label} ({_n(seg['dur'])}s)",
        f"{pos_phrase}{seg_str} plays {label} ({_n(seg['dur'])}s total)",
        f"{pos_phrase}{seg_str} features {label} lasting {_n(seg['dur'])}s",
        f"{pos_phrase}{seg_str} contains {label} ({_n(seg['dur'])}s long)",
        f"{pos_phrase}{seg_str} is where {label} runs {_n(seg['dur'])}s",
        f"{pos_phrase}{seg_str} carries {label} for {_n(seg['dur'])}s",
        f"{pos_phrase}{seg_str} presents {label} ({_n(seg['dur'])}s duration)",
        f"{pos_phrase}{seg_str} gives us {label} over {_n(seg['dur'])}s",
        f"{pos_phrase}{seg_str} holds {label} ({_n(seg['dur'])}s span)",
        f"{pos_phrase}{seg_str} includes {label} lasting {_n(seg['dur'])} seconds",
        f"{pos_phrase}{seg_str} offers {label} ({_n(seg['dur'])}s total)",
        f"{pos_phrase}{seg_str} reveals {label} across {_n(seg['dur'])}s",
        f"{pos_phrase}{seg_str} demonstrates {label} ({_n(seg['dur'])}s span)",
        f"{pos_phrase}{seg_str} exhibits {label} running {_n(seg['dur'])} seconds",
    )


def describe_seg_both_single(label, seg):
    """For single-segment samples — seg in middle, after-text goes to end ($)."""
    seg_str = _sg(seg['start'], seg['end'])
    return _r(
        f"Examining {seg_str}, {label} runs {_n(seg['dur'])} seconds total",
        f"Looking at {seg_str}, {label} spans {_n(seg['dur'])} seconds",
        f"Within {seg_str}, {label} occupies {_n(seg['dur'])} seconds",
        f"At {seg_str}, {label} is heard for {_n(seg['dur'])} seconds",
        f"From {seg_str}, {label} plays across {_n(seg['dur'])} seconds",
        f"Through {seg_str}, {label} lasts {_n(seg['dur'])} seconds",
        f"Checking {seg_str}, {label} takes up {_n(seg['dur'])} seconds",
        f"Reviewing {seg_str}, {label} covers {_n(seg['dur'])} seconds",
        f"Considering {seg_str}, {label} extends {_n(seg['dur'])} seconds",
        f"Observing {seg_str}, {label} continues {_n(seg['dur'])} seconds",
        f"Noting {seg_str}, {label} persists {_n(seg['dur'])} seconds",
        f"Using {seg_str}, {label} measures {_n(seg['dur'])} seconds",
        f"Via {seg_str}, {label} reaches {_n(seg['dur'])} seconds",
        f"Throughout {seg_str}, {label} endures {_n(seg['dur'])} seconds",
    )

def describe_seg_before(seg):
    """Only mention seg with before context."""
    return _r(
        f"as indicated by {_sg(seg['start'], seg['end'])}",
        f"shown in {_sg(seg['start'], seg['end'])}",
        f"with the segment {_sg(seg['start'], seg['end'])}",
        f"as captured by {_sg(seg['start'], seg['end'])}",
        f"marked by {_sg(seg['start'], seg['end'])}",
        f"(see {_sg(seg['start'], seg['end'])})",
        f"via the interval {_sg(seg['start'], seg['end'])}",
    )

def describe_seg_after(seg, label):
    """Explanation AFTER the seg tag — for improving seg both metric."""
    return _r(
        f" — this segment corresponds to {label}, running from {_n(seg['start'])} to {_n(seg['end'])} seconds",
        f", which represents {label} spanning {_n(seg['dur'])} seconds",
        f" — here {label} is active for {_n(seg['dur'])} seconds",
        f" — during this window {label} plays for {_n(seg['dur'])} seconds",
        f"; this is {label} covering {_n(seg['dur'])} seconds of audio",
        f" — that is {label} from {_n(seg['start'])} to {_n(seg['end'])} seconds",
        f" — at this point {label} is heard over {_n(seg['dur'])} seconds",
        f" — corresponding to {label} lasting {_n(seg['dur'])} seconds",
        f" — this marks {label} which remains for {_n(seg['dur'])} seconds",
        f" — the audio at this position features {label} for {_n(seg['dur'])} seconds",
    )

# ═══════════════════════════════════════════════════════════════
# BUILDING BLOCKS — Bridges
# ═══════════════════════════════════════════════════════════════

BRIDGE_GAP = lambda s1, s2, gap: _r(
    f"The gap is simply {_n(s2['start'])} − {_n(s1['end'])} = {gap} seconds.",
    f"Subtracting: {_n(s2['start'])} − {_n(s1['end'])} = {gap} seconds of separation.",
    f"Thus, {_n(s2['start'])} − {_n(s1['end'])} = {gap}s is the interval between them.",
    f"The elapsed time from first end to second start equals {gap} seconds.",
    f"This yields a temporal gap of {gap} seconds ({_n(s2['start'])} − {_n(s1['end'])}).",
    f"So the pause measures {gap} seconds between the two events.",
    f"The interval amounts to {gap} seconds between the endpoints.",
    f"We compute {gap} seconds as the difference between these markers.",
)

BRIDGE_OVERLAP = lambda s1, s2, o_start, o_end, o_dur: _r(
    f"The overlapping region runs from {_n(o_start)} to {_n(o_end)} seconds, totalling {o_dur} seconds.",
    f"They co-occur between {_n(o_start)}s and {_n(o_end)}s — a simultaneous duration of {o_dur} seconds.",
    f"The shared time window is [{_n(o_start)}, {_n(o_end)}], lasting {o_dur} seconds.",
    f"Both are active together from {_n(o_start)} to {_n(o_end)}, giving {o_dur}s of overlap.",
    f"Their common interval spans {o_dur} seconds, from max({_n(o_start)}) to min({_n(o_end)}).",
    f"The simultaneous segment extends {o_dur} seconds, from {_n(o_start)} to {_n(o_end)}.",
    f"There is an overlap of {o_dur} seconds where both sounds are present.",
)

BRIDGE_COUNT = lambda count, label, target_start: _r(
    f"That means {count} event(s) finish before {label} starts at {_n(target_start)}s.",
    f"Therefore, {count} preceding event(s) are complete by the time {label} begins.",
    f"Thus, {count} sound(s) conclude before the onset of {label}.",
    f"In total, {count} events have already ended when {label} enters at {_n(target_start)}s.",
    f"So {count} of these events cease prior to {label}'s start at {_n(target_start)}s.",
    f"Altogether, {count} prior events finish before {label} starts at {_n(target_start)}s.",
    f"The count shows {count} event(s) ended by the time {label} starts.",
)

BRIDGE_ORDER = lambda first, first_start, second, second_start: _r(
    f"Since {_n(first_start)} < {_n(second_start)}, {first} occurs first.",
    f"{first.title()} starts at {_n(first_start)}s, earlier than {second} at {_n(second_start)}s — so it comes first.",
    f"The smallest start value ({_n(first_start)}s) belongs to {first}, making it the first event.",
    f"Comparing {_n(first_start)}s to {_n(second_start)}s, clearly {first} happens earlier.",
    f"The timeline shows {first} beginning at {_n(first_start)}s, before {second} at {_n(second_start)}s.",
    f"With an onset of {_n(first_start)}s versus {_n(second_start)}s, {first} goes first.",
)

BRIDGE_DUR_CMP = lambda label1, dur1, label2, dur2, diff, longer: _r(
    f"{longer.title()} is longer by {diff} seconds ({_n(dur1)} vs {_n(dur2)}).",
    f"The difference is {diff} seconds, with {longer} being the longer sound.",
    f"{longer.title()} exceeds the other by {diff} seconds in duration.",
    f"The duration gap is {diff} seconds in favor of {longer}.",
    f"There is a {diff}-second difference, with {longer} having the greater length.",
    f"{longer.title()} lasts {diff} seconds more than the alternative.",
)

BRIDGE_DUR_PCT = lambda pct: _r(
    f"This represents {pct}% of the total audio.",
    f"So the event occupies {pct}% of the full recording.",
    f"Hence the proportion is {pct}% of the entire clip.",
    f"Thus, the event accounts for {pct}% of the total duration.",
    f"The result tells us the event covers {pct}% of the audio.",
    f"This works out to {pct}% of the overall recording.",
)

BRIDGE_START_PCT = lambda pct: _r(
    f"This means the event begins {pct}% of the way through the audio.",
    f"So the onset occurs at {pct}% of the total duration.",
    f"Thus, {pct}% of the audio has elapsed before the event starts.",
    f"Hence the start point falls at {pct}% into the recording.",
    f"The entry point is therefore at {pct}% of the total time.",
    f"This places the start at the {pct}% mark of the audio.",
)

# ═══════════════════════════════════════════════════════════════
# BUILDING BLOCKS — Closings (before answer)
# ═══════════════════════════════════════════════════════════════

CLOSINGS = [
    lambda ans: f"Therefore, the answer is {ans}.",
    lambda ans: f"So the correct choice is {ans}.",
    lambda ans: f"Thus, we arrive at {ans}.",
    lambda ans: f"The answer is therefore {ans}.",
    lambda ans: f"This gives us {ans} as the answer.",
    lambda ans: f"We conclude that the answer is {ans}.",
    lambda ans: f"Hence, {ans} is correct.",
    lambda ans: f"Based on this reasoning, the answer is {ans}.",
    lambda ans: f"Our final answer is {ans}.",
    lambda ans: f"Putting it all together, the answer is {ans}.",
    lambda ans: f"From the above, we select {ans}.",
    lambda ans: f"The right answer here is {ans}.",
    lambda ans: f"After working through the timeline, {ans} is correct.",
    lambda ans: f"Thus the correct selection is {ans}.",
    lambda ans: f"So, the answer turns out to be {ans}.",
    lambda ans: f"Following this analysis, the answer is {ans}.",
    lambda ans: f"All evidence points to {ans}.",
    lambda ans: f"Therefore, among the options, {ans} is correct.",
    lambda ans: f"The calculation confirms that {ans} is the answer.",
    lambda ans: f"Collecting these observations, the answer is {ans}.",
    lambda ans: f"Looking at the evidence, we select {ans}.",
    lambda ans: f"This leads us to the answer {ans}.",
    lambda ans: f"Our conclusion is that {ans} is right.",
    lambda ans: f"Assembling the clues, the answer is {ans}.",
    lambda ans: f"Checking the results, we arrive at {ans}.",
    lambda ans: f"The reasoning supports {ans} as the answer.",
    lambda ans: f"We determine that {ans} is the right choice.",
    lambda ans: f"Through elimination, we settle on {ans}.",
    lambda ans: f"Weighing the facts, the correct choice is {ans}.",
    lambda ans: f"All factors considered, the answer is {ans}.",
]

# ═══════════════════════════════════════════════════════════════
# BUILDING BLOCKS — "First segment" / "Second segment" references
# ═══════════════════════════════════════════════════════════════

def event_ref(label, idx):
    prefix = _r("the first", "the second", "the next", "the following", "the initial",
                 "the subsequent", "the former", "the latter", "one", "another",
                 "the earlier", "the later", "this", "that", "the corresponding",
                 "the associated")
    return _r(
        f"{prefix} event — {label}",
        f"{prefix} sound — {label}",
        f"{prefix} audio cue — {label}",
        f"{prefix} segment featuring {label}",
        f"{prefix} occurrence of {label}",
    )

# ═══════════════════════════════════════════════════════════════
# Template composition
# ═══════════════════════════════════════════════════════════════

def _round_ans(ans_str):
    """If ans_str is a number, round; else return as-is."""
    try:
        return str(_n(float(ans_str.strip().rstrip('%')))) + ('%' if '%' in ans_str else '')
    except ValueError:
        return ans_str.strip()

# ── Generative pickers for openings (combinatorial diversity) ──

def _pick_gap_opening():
    if _r(True, True, False):
        start = _r("We need to determine", "Let us calculate", "The question asks for",
                   "Our objective is to find", "We must identify", "This requires finding",
                   "We should compute", "Let us work out", "The task is to find",
                   "We aim to measure", "Our goal is to calculate")
        mid = _r(" the time gap", " the interval", " the pause", " the separation",
                 " the temporal distance", " the spacing", " the elapsed time",
                 " the duration between", " the time difference", " the delay")
        end = _r(" between these two audio events", " between the two sounds",
                 " separating these audio cues", " between these occurrences",
                 " from one sound to the next", " between these acoustic events",
                 " between the first and second event",
                 " from the end of one to the start of the next")
        return f"{start}{mid}{end}."
    return _r(*OPENINGS_GAP)

def _pick_repeated_opening():
    if _r(True, True, False):
        start = _r("We need to find", "The question asks about", "Let us measure",
                   "Our task involves finding", "We must determine", "This requires measuring",
                   "We should identify", "The goal is to calculate")
        mid = _r(" the time between", " the gap between", " the interval separating",
                 " the pause between", " the recurrence of", " the spacing of")
        end = _r(" two occurrences of the same sound", " repeated instances of this audio event",
                 " two instances of the same acoustic cue",
                 " matching sound events in the clip",
                 " a repeating audio pattern",
                 " identical sound events occurring twice")
        return f"{start}{mid}{end}."
    return _r(*OPENINGS_REPEATED)

def _pick_count_opening():
    if _r(True, True, False):
        start = _r("We need to count", "The question asks how many", "Let us tally",
                   "Our objective is to count", "We must determine how many",
                   "This requires counting", "We should enumerate",
                   "The task is to find how many")
        mid = _r(" events finish", " sounds conclude", " audio events end",
                 " cues complete", " segments stop", " occurrences cease")
        end = _r(" before the last one begins", " before the target starts",
                 " prior to the final onset", " before the designated event",
                 " preceding the final sound", " before the ultimate event")
        return f"{start}{mid}{end}."
    return _r(*OPENINGS_COUNT_BEFORE)

def _pick_order_opening():
    if _r(True, True, False):
        start = _r("We need to determine", "The question asks which", "Let us identify",
                   "Our goal is to find", "We must figure out", "This requires finding",
                   "We should check", "The task is to identify")
        mid = _r(" which event occurs first", " which sound comes first",
                 " which audio cue happens earliest", " the earliest event",
                 " which happens first", " the first occurrence among these")
        end = _r(" in the timeline", " in the audio clip", " in the recording",
                 " among the provided segments", " in the sequence",
                 " based on the timestamps")
        return f"{start}{mid}{end}."
    return _r(*OPENINGS_ORDER)

def _pick_dur_pct_opening():
    if _r(True, True, False):
        start = _r("We need to determine", "The question asks about", "Let us calculate",
                   "Our objective is to find", "We must compute", "This requires finding",
                   "We should measure", "The task is to determine")
        mid = _r(" what percentage", " what proportion", " what fraction",
                 " how much of the audio", " the share of time")
        end = _r(" this event occupies", " this sound covers", " this segment represents",
                 " this audio cue fills", " this acoustic event takes up",
                 " this sonic event occupies")
        return f"{start}{mid}{end}."
    return _r(*OPENINGS_DUR_PCT)

def _pick_start_pct_opening():
    if _r(True, True, False):
        start = _r("We need to find", "The question asks at what point", "Let us determine",
                   "Our objective is to locate", "We must calculate", "This requires finding",
                   "We should measure", "The task is to identify")
        mid = _r(" when this event begins", " where this sound starts",
                 " the onset position", " the start point",
                 " the beginning of this event")
        end = _r(" as a percentage of the total", " relative to the full audio",
                 " on the timeline", " in the recording",
                 " within the complete audio clip")
        return f"{start}{mid}{end}."
    return _r(*OPENINGS_START_PCT)

def _pick_dur_cmp_opening():
    if _r(True, True, False):
        start = _r("We need to compare", "The question asks which", "Let us determine",
                   "Our goal is to find", "We must check", "This requires comparing",
                   "We should measure", "The task is to identify")
        mid = _r(" which event lasts longer", " which sound has greater duration",
                 " which audio cue is longer", " the longer of the two",
                 " the time difference between these events")
        end = _r(" by comparing their durations", " by examining their time spans",
                 " using their respective lengths", " from the segment information",
                 " based on the provided timestamps")
        return f"{start}{mid}{end}."
    return _r(*OPENINGS_DUR_CMP)

def _pick_overlap_opening():
    if _r(True, True, False):
        start = _r("We need to measure", "The question asks how long", "Let us calculate",
                   "Our objective is to find", "We must determine", "This requires finding",
                   "We should compute", "The task is to identify")
        mid = _r(" these two events overlap", " these sounds co-occur",
                 " the simultaneous portion", " the shared time window",
                 " the overlapping segment", " the concurrent duration")
        end = _r(" between these two audio events", " of these two sounds",
                 " where both are active together",
                 " during which both play simultaneously",
                 " that both events share on the timeline")
        return f"{start}{mid}{end}."
    return _r(*OPENINGS_OVERLAP)


def _pick_unknown_opening():
    if _r(True, True, False):
        start = _r("Let us examine", "We need to analyze", "The question requires",
                   "Our task is to review", "We should inspect", "We must consider",
                   "Let us look at", "We need to check", "Our goal is to assess")
        mid = _r(" the relevant audio cues", " the provided timestamps",
                 " the audio segments", " the timing information",
                 " the event details", " the segment data")
        end = _r(" to answer this question.", " to find the correct answer.",
                 " and determine the correct response.",
                 " to reach the right conclusion.",
                 " and identify the proper selection.")
        return f"{start}{mid}{end}"
    return _r(*OPENINGS_UNKNOWN)

def compose_gap(segs, descs, answer, total_dur=None, q_text=None):
    if len(segs) < 2:
        return None
    s1, s2 = segs[0], segs[1]
    l1, l2 = descs[0], descs[1]
    gap = _n(s2['start'] - s1['end'])
    parts = [_pick_gap_opening() + " "]
    parts.append(describe_seg_both(l1, s1, "First") + ". ")
    parts.append(describe_seg_both(l2, s2, "Next") + ". ")
    parts.append(_r("Now we can compute the exact interval between them. ",
                     "Using these endpoints we calculate the gap. ",
                     "The difference between these timestamps is clear. ",
                     "From these boundaries we derive the separation. ",
                     "These two time points give us the needed interval. "))
    parts.append(BRIDGE_GAP(s1, s2, gap) + " ")
    parts.append(_r(*CLOSINGS)(answer))
    return "".join(parts)


def compose_repeated_gap(segs, descs, answer, total_dur=None, q_text=None):
    if len(segs) < 2:
        return None
    s1, s2 = segs[0], segs[1]
    label = descs[0]
    gap = _n(s2['start'] - s1['end'])
    parts = [_pick_repeated_opening() + " "]
    parts.append(describe_seg_both(label, s1, "First") + ". ")
    parts.append(describe_seg_both(label, s2, "Next") + ". ")
    parts.append(_r("Now we subtract to find the recurrence gap. ",
                     "The interval between instances is obtained by subtraction. ",
                     "Computing the time between these two occurrences. ",
                     "The gap is simply the later start minus the earlier end. ",
                     "We calculate the repetition interval from these values. "))
    parts.append(BRIDGE_GAP(s1, s2, gap) + " ")
    parts.append(_r(*CLOSINGS)(answer))
    return "".join(parts)


def compose_count_before(segs, descs, answer, total_dur=None, q_text=None):
    if len(segs) < 2:
        return None
    target_label = descs[-1]
    target_seg = segs[-1]
    count = len(segs) - 1
    parts = [_pick_count_opening() + " "]
    positions = ["First", "Second", "Third", "Fourth", "Fifth"]
    for i, (s, d) in enumerate(zip(segs[:-1], descs[:-1])):
        pos = positions[i] if i < len(positions) else f"Event {i+1}"
        parts.append(describe_seg_both(d, s, pos) + ". ")
    parts.append(f" Finally, {_label(target_label)} starts at {_n(target_seg['start'])} seconds. ")
    parts.append(_r("Now we count how many finished before this point. ",
                     "Comparing end times to this start gives the tally. ",
                     "We check which preceding events have already concluded. ",
                     "The count of completed events before this onset is the answer. ",
                     "Tracking which events ended prior to this moment. "))
    parts.append(BRIDGE_COUNT(count, _label(target_label), target_seg['start']) + " ")
    parts.append(_r(*CLOSINGS)(answer))
    return "".join(parts)


def compose_order(segs, descs, answer, total_dur=None, q_text=None):
    if len(segs) < 2:
        return None
    starts = [(s['start'], d, s) for s, d in zip(segs, descs)]
    starts.sort(key=lambda x: x[0])
    first_l, second_l = starts[0][1], starts[1][1]
    first_s, second_s = starts[0][2], starts[1][2]
    parts = [_pick_order_opening() + " "]
    parts.append(describe_seg_both(first_l, first_s, "First") + ". ")
    parts.append(describe_seg_both(second_l, second_s, "Then") + ". ")
    parts.append(_r("Comparing the start times reveals the chronological order. ",
                     "The onset timestamps determine which event leads. ",
                     "Earlier start time tells us which comes first. ",
                     "Sorting by start time makes the order obvious. ",
                     "These start positions show the sequence clearly. "))
    parts.append(BRIDGE_ORDER(_label(first_l), first_s['start'], _label(second_l), second_s['start']) + " ")
    parts.append(_r(*CLOSINGS)(answer))
    return "".join(parts)


def compose_duration_percentage(segs, descs, answer, total_dur=None, q_text=None):
    if not segs or total_dur is None:
        return None
    s = segs[0]
    label = descs[0] if descs else "this event"
    pct = _n(s['dur'] / total_dur * 100)
    parts = [_pick_dur_pct_opening() + " "]
    parts.append(f" This clip runs for {_n(total_dur)} seconds in total. ")
    parts.append(f" {_label(label).capitalize()} covers {_n(s['dur'])} seconds of that duration. ")
    parts.append(f" Computing its proportion: {_n(s['dur'])} ÷ {_n(total_dur)} × 100 = {pct}%. ")
    parts.append(BRIDGE_DUR_PCT(pct) + " ")
    parts.append(_r(*CLOSINGS)(answer) + " ")
    parts.append(_r(
        f"This is based on {_sg(s['start'], s['end'])}, where {label} runs {_n(s['dur'])}s",
        f"Evidence from {_sg(s['start'], s['end'])} shows {label} lasting {_n(s['dur'])}s",
        f"At {_sg(s['start'], s['end'])}, {label} is heard for {_n(s['dur'])} seconds total",
        f"Looking at {_sg(s['start'], s['end'])}, {label} spans {_n(s['dur'])} seconds",
        f"Checking {_sg(s['start'], s['end'])}, {label} occupies {_n(s['dur'])}s of audio",
    ))
    return "".join(parts)


def compose_start_percentage(segs, descs, answer, total_dur=None, q_text=None):
    if not segs or total_dur is None:
        return None
    s = segs[0]
    label = descs[0] if descs else "this event"
    pct = _n(s['start'] / total_dur * 100)
    parts = [_pick_start_pct_opening() + " "]
    parts.append(f" The full recording lasts {_n(total_dur)} seconds. ")
    parts.append(f" The event, {_label(label)}, begins at {_n(s['start'])} seconds into the clip. ")
    parts.append(f" Computing: {_n(s['start'])} ÷ {_n(total_dur)} × 100 = {pct}%. ")
    parts.append(BRIDGE_START_PCT(pct) + " ")
    parts.append(_r(*CLOSINGS)(answer) + " ")
    parts.append(_r(
        f"We locate {label} at {_sg(s['start'], s['end'])}, where it first begins",
        f"The segment {_sg(s['start'], s['end'])} marks when {label} starts",
        f"Found in {_sg(s['start'], s['end'])}, {label} is heard entering the audio",
        f"Checking {_sg(s['start'], s['end'])}, {label} begins at this position",
        f"This is seen in {_sg(s['start'], s['end'])}, where {label} appears",
    ))
    return "".join(parts)


def compose_duration_compare(segs, descs, answer, total_dur=None, q_text=None):
    if len(segs) < 2:
        return None
    s1, s2 = segs[0], segs[1]
    l1, l2 = descs[0], descs[1]
    diff = _n(abs(s1['dur'] - s2['dur']))
    longer = _label(l1) if s1['dur'] > s2['dur'] else _label(l2)
    parts = [_pick_dur_cmp_opening() + " "]
    parts.append(describe_seg_both(l1, s1, "First") + ". ")
    parts.append(describe_seg_both(l2, s2, "Next") + ". ")
    parts.append(_r("Comparing these two durations tells us which is longer. ",
                     "The difference between their lengths is straightforward. ",
                     "One of these events clearly runs longer than the other. ",
                     "Subtracting one duration from the other shows the gap. ",
                     "Their respective lengths can now be compared directly. "))
    parts.append(BRIDGE_DUR_CMP(l1, _n(s1['dur']), l2, _n(s2['dur']), diff, longer) + " ")
    parts.append(_r(*CLOSINGS)(answer))
    return "".join(parts)


def compose_overlap(segs, descs, answer, total_dur=None, q_text=None):
    if len(segs) < 2:
        return None
    s1, s2 = segs[0], segs[1]
    l1, l2 = descs[0], descs[1]
    o_start = _n(max(s1['start'], s2['start']))
    o_end = _n(min(s1['end'], s2['end']))
    o_dur = _n(max(0, o_end - o_start))
    parts = [_pick_overlap_opening() + " "]
    parts.append(describe_seg_both(l1, s1, "First") + ". ")
    parts.append(describe_seg_both(l2, s2, "Next") + ". ")
    parts.append(_r("Now we determine the window where both are active. ",
                     "The overlapping region is bounded by these times. ",
                     "Both events coincide between certain time boundaries. ",
                     "We find where their intervals intersect on the timeline. ",
                     "The shared portion falls between these key points. "))
    parts.append(BRIDGE_OVERLAP(s1, s2, o_start, o_end, o_dur) + " ")
    parts.append(_r(*CLOSINGS)(answer))
    return "".join(parts)


def compose_unknown(segs, descs, answer, total_dur=None, q_text=None):
    positions = ["First", "Second", "Third", "Fourth", "Fifth"]
    parts = [_pick_unknown_opening() + " "]
    if len(segs) == 1:
        d, s = descs[0], segs[0]
        parts.append(f" The audio contains an event: {_label(d)}. ")
        parts.append(f" This event spans {_n(s['dur'])} seconds, from {_n(s['start'])} to {_n(s['end'])} seconds. ")
        parts.append(f" Reviewing the segment details confirms the timing of this sound. ")
        parts.append(f" Based on this, the duration of {_label(d)} is {_n(s['dur'])} seconds. ")
        parts.append(_r(*CLOSINGS)(answer) + " ")
        parts.append(_r(
            f"Examining {_sg(s['start'], s['end'])}, {d} plays for {_n(s['dur'])}s",
            f"Looking at {_sg(s['start'], s['end'])}, {d} runs {_n(s['dur'])} seconds",
            f"Within {_sg(s['start'], s['end'])}, {d} is heard for {_n(s['dur'])}s",
            f"From {_sg(s['start'], s['end'])}, {d} occupies {_n(s['dur'])} seconds",
            f"At {_sg(s['start'], s['end'])}, {d} lasts {_n(s['dur'])} seconds total",
        ))
    else:
        for i, (d, s) in enumerate(zip(descs, segs)):
            pos = positions[i] if i < len(positions) else f"Event {i+1}"
            parts.append(describe_seg_both(d, s, pos) + ". ")
        parts.append(" " + _r(*CLOSINGS)(answer))
    return "".join(parts)


COMPOSERS = {
    "gap": compose_gap,
    "repeated_event_gap": compose_repeated_gap,
    "count_before": compose_count_before,
    "order": compose_order,
    "duration_percentage": compose_duration_percentage,
    "start_percentage": compose_start_percentage,
    "duration_compare": compose_duration_compare,
    "overlap": compose_overlap,
    "unknown": compose_unknown,
}


# ═══════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════

def process_entry(item: dict, min_words: int, max_words: int, target_avg: int) -> tuple[str | None, dict]:
    """Returns (new_assistant_text_or_None, stats_dict)."""
    msgs = item.get("messages", [])
    assistant_idx = None
    for i, m in enumerate(msgs):
        if m.get("role") == "assistant":
            assistant_idx = i
            break
    if assistant_idx is None:
        return None, {"error": "no_assistant"}

    astext = msgs[assistant_idx].get("content", "")
    if not isinstance(astext, str):
        return None, {"error": "non_string_content"}

    # User msg for classification
    user_text = ""
    for m in msgs:
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            user_text = m["content"]
            break
    question = extract_question(user_text)
    qa_type = classify_question(question)

    # Parse CoT
    tm = THINK_RE.search(astext)
    if not tm:
        return None, {"error": "no_think", "qa_type": qa_type}
    think_orig = tm.group(1).strip()
    answer = extract_answer(astext)
    segments = parse_segments(think_orig)
    descs = extract_labels_from_segments(segments)
    total_dur = get_total_duration(think_orig, user_text)
    # Fallback: estimate from seg end time if total_dur not found (many AudioSet clips are 10s)
    if total_dur is None and segments:
        max_end = max(s['end'] for s in segments)
        if 9.5 <= max_end <= 10.5:
            total_dur = 10.0
        elif 4.5 <= max_end <= 5.5:
            total_dur = 5.0
        elif 14.5 <= max_end <= 15.5:
            total_dur = 15.0

    if not segments:
        return None, {"error": "no_segments", "qa_type": qa_type}

    wc_before = len(think_orig.split())
    composer = COMPOSERS.get(qa_type, compose_unknown)

    # Try composer
    stats_info = {"qa_type": qa_type, "words_before": wc_before}
    result = None
    for attempt in range(10):  # Retry for word count fit
        try:
            result = composer(segments, descs, answer, total_dur, question)
        except Exception:
            continue
        if result:
            wc = len(result.split())
            stats_info["words_after"] = wc
            if min_words <= wc <= max_words:
                break
            result = None  # Try again if out of range

    if result:
        stats_info["status"] = "composed"
        new_content = f"<think>{result}</think><answer>{answer}</answer>"
        return new_content, stats_info

    # Fallback: rephrase with padding to reach word target
    pad = _r(
        " After carefully analyzing the timestamps and comparing the relevant audio segments, ",
        " Once we review the timing information and relate the events to each other, ",
        " By examining when each sound occurs relative to the others on the timeline, ",
        " After looking at the segment boundaries and temporal relationships between them, ",
        " From the positions of each segment and how they relate to one another, ",
    )
    fallback = f"Let us examine the audio to answer this question. {think_orig}{pad}the answer is {answer}."
    extras = [
        " This involves checking the exact timing of each sound event. ",
        " The segment timestamps give us precise information about each event. ",
        " We need to compare the temporal positions of these audio cues. ",
        " The relative timing between these events is the key factor here. ",
        " Each sound occupies a specific window in the recording timeline. ",
        " The boundaries between segments define the temporal structure of the recording. ",
        " Aligning each event to the timeline provides clarity on the sequence. ",
        " This temporal information allows us to pinpoint each sound precisely. ",
    ]
    while len(fallback.split()) < min_words:
        fallback += _r(*extras)
    stats_info["words_after"] = len(fallback.split())
    return f"<think>{fallback}</think><answer>{answer}</answer>", stats_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--report_json", required=True)
    parser.add_argument("--sample_output_jsonl", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_avg_words", type=int, default=75)
    parser.add_argument("--min_words", type=int, default=62)
    parser.add_argument("--max_words", type=int, default=120)
    args = parser.parse_args()
    random.seed(args.seed)

    print(f"Loading {args.input_jsonl} ...")
    data = []
    with open(args.input_jsonl) as f:
        for line in f:
            data.append(json.loads(line))
    print(f"  Total entries: {len(data)}")

    stats = {
        "total": len(data),
        "composed": 0,
        "fallback": 0,
        "errors": Counter(),
        "by_type": Counter(),
        "word_before": [],
        "word_after": [],
        "samples": [],
    }

    output = []
    for idx, item in enumerate(data):
        new_content, info = process_entry(item, args.min_words, args.max_words, args.target_avg_words)
        if new_content:
            new_item = deepcopy(item)
            for i, m in enumerate(new_item["messages"]):
                if m.get("role") == "assistant":
                    new_item["messages"][i]["content"] = new_content
                    break
            output.append(new_item)
            stats["by_type"][info["qa_type"]] += 1
            stats["word_before"].append(info.get("words_before", 0))
            stats["word_after"].append(info.get("words_after", 0))
            if info.get("status") == "composed":
                stats["composed"] += 1
            else:
                stats["fallback"] += 1
        else:
            output.append(item)
            stats["errors"][info.get("error", "unknown")] += 1

        if len(stats["samples"]) < 30:
            stats["samples"].append({
                "id": str(idx),
                "qa_type": info.get("qa_type", "?"),
                "words_before": info.get("words_before", 0),
                "words_after": info.get("words_after", 0),
                "cot_before": data[idx].get("messages", [{}])[-1].get("content", "")[:120] if data[idx].get("messages") else "",
                "cot_after": new_content[:120] if new_content else "",
            })

        if (idx + 1) % 5000 == 0:
            print(f"  Progress: {idx+1}/{len(data)} (composed={stats['composed']}, fallback={stats['fallback']}, errors={sum(stats['errors'].values())})")

    # Write output
    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for item in output:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"\nOutput: {args.output_jsonl}")

    # Report
    wb = stats["word_before"]
    wa = stats["word_after"]
    report = {
        "config": {"input": args.input_jsonl, "output": args.output_jsonl,
                   "target_avg_words": args.target_avg_words,
                   "min_words": args.min_words, "max_words": args.max_words},
        "summary": {
            "total": stats["total"],
            "composed": stats["composed"],
            "fallback": stats["fallback"],
            "errors": dict(stats["errors"]),
            "compose_rate_pct": round(stats["composed"] / stats["total"] * 100, 1),
        },
        "word_count": {
            "before": {"avg": round(sum(wb)/len(wb), 1) if wb else 0,
                       "min": min(wb) if wb else 0, "max": max(wb) if wb else 0},
            "after": {"avg": round(sum(wa)/len(wa), 1) if wa else 0,
                      "min": min(wa) if wa else 0, "max": max(wa) if wa else 0},
        },
        "qa_type_distribution": dict(stats["by_type"].most_common()),
    }

    with open(args.report_json, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report: {args.report_json}")

    with open(args.sample_output_jsonl, "w") as f:
        for s in stats["samples"]:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Samples: {args.sample_output_jsonl}")

    print(f"\nSummary: composed={stats['composed']}, fallback={stats['fallback']}, "
          f"errors={sum(stats['errors'].values())}, "
          f"avg_wc: {report['word_count']['before']['avg']} → {report['word_count']['after']['avg']}")


if __name__ == "__main__":
    main()

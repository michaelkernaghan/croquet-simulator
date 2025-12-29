# Reference Documents Guide for Croquet Simulator

You have access to croquet reference documents in `reference_docs/` that should inform the simulator's tactical and strategic decision-making:

## Document Structure

### JSON Content (machine-readable, preferred)

- `wylie/expert-croquet-tactics/` - Keith Wylie's "Expert Croquet Tactics" - the definitive guide to Association Croquet strategy including:
  - Triple peel execution and timing
  - Break organization and priorities
  - Opening theory and standard leaves
  - Shot selection and risk assessment

- `wylie/advanced-play/` - Advanced techniques and shot-making
- `wylie/practice-and-training/` - Training drills and skill development
- `wylie/tournament-play/` - Tournament-level strategy

- `oxford-croquet/` - Strategic articles from Oxford Croquet covering tactics, technique, and game theory

- `coaching/` - Coaching handbook with fundamentals and shot technique

- `commentary/` - Real game commentary showing decision-making in practice

### Text Summaries (USE THESE - Machine Readable)

- `basic-ac-tactics.txt` - **START HERE** - Comprehensive guide to AC tactics including:
  - Shot types (drives, rolls, stop shots, splits, takeoffs)
  - Break building (2-ball, 3-ball, 4-ball breaks)
  - Opening tactics (tices, supershot, responses)
  - Leaves (DSL, NSL, defensive spreads)
  - Tactical decision-making

- `pdfs/ac-laws-summary.txt` - Key AC rules for the simulator

### PDFs (DO NOT READ DIRECTLY - TOO LARGE)

**WARNING**: The PDF files are too large for direct reading. Use the text summaries above instead.

- `pdfs/AC-Laws-Rulings-Commentary-Combined.pdf` - (original PDF - do not read)
- `pdfs/croquetcoachinghandbook.pdf` - (original PDF - do not read)
- `pdfs/WCF-GC-Rules-6th-Edition-Final-7.3.22.pdf` - (original PDF - do not read)

## How to Use These Documents

1. **For AI opponent decision-making**: Reference Wylie's tactical priorities, break organization, and leave selection
2. **For shot evaluation**: Use the coaching content for shot difficulty and technique factors
3. **For legal move validation**: Consult AC Laws for rules on faults, lifts, and valid plays
4. **For realistic gameplay**: Commentary shows how experts weigh options in real situations

## JSON File Format

Each JSON file contains:
```json
{
  "url": "original source URL",
  "title": "page title",
  "content": "full text content - USE THIS FOR CONTEXT",
  "structure": {
    "paragraphs": [...],
    "headers": [...],
    "images": [{ "localPath": "images/..." }]
  },
  "links": [...]
}
```

The `content` field contains the full text - use this for context when implementing tactical logic.

## Key Concepts from Wylie (Expert Croquet Tactics)

### Break Organization
- Priorities for maintaining a 4-ball break
- Pioneer placement and escape ball management
- When to take risks vs play safe

### Triple Peel Timing
- 4-back peel positions (before/after hoops 3, 5, 6, 1-back)
- Penult peel timing and continuation
- Rover peel and peg-out execution

### Opening Theory
- Standard opening sequences
- Tice placement and responses
- When to shoot vs when to lay up

### Leaves
- New Standard Leave (NSL)
- Defensive 4-back leaves
- Leave selection based on game state

### Style Categories (from Article 3)
- Aggressive croquet - high risk/reward
- Precision croquet - technical accuracy
- Canny croquet - defensive, percentage play
- Monte Carlo croquet - calculated gambling

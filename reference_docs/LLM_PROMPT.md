# Prompt for Building the Croquet Simulator

You are building an Association Croquet simulator. Use the reference documents in this folder to create accurate, realistic gameplay. Here's how to get the most from these resources:

## Quick Start - Read These First

1. **`basic-ac-tactics.txt`** - Your primary reference for gameplay logic:
   - Shot types and their ratios (drives=1:3, stop shots=1:5-1:10, rolls=1:2)
   - Break patterns (how to chain multiple hoops in one turn)
   - Opening sequences (first 4 turns of the game)
   - Leaves (end-of-turn ball positioning strategies)

2. **`pdfs/ac-laws-summary.txt`** - Rules reference:
   - Legal moves and turn structure
   - Hoop running requirements
   - Faults and penalties
   - Ball state tracking (live/dead balls)

## Deep Dive - JSON Content

For detailed tactical content, read the JSON files in these folders. Each JSON has a `content` field with the full text:

- **`wylie/expert-croquet-tactics/`** - Advanced strategy from Keith Wylie's book
- **`wylie/advanced-play/`** - Advanced techniques
- **`coaching/`** - Fundamentals and shot technique
- **`oxford-croquet/`** - Strategic articles

## Key Concepts to Implement

### 1. Turn Structure
```
START TURN
  ├── Strike ball once
  ├── If ROQUET (hit another ball):
  │     ├── Place balls in contact
  │     ├── Play CROQUET STROKE (both balls move)
  │     └── Play CONTINUATION STROKE
  ├── If RUN HOOP:
  │     ├── Score point
  │     ├── Reset all balls to "live"
  │     └── Play CONTINUATION STROKE
  └── If NEITHER: END TURN
```

### 2. Ball States
Track for each ball:
- Current hoop (1-12, or rover, or pegged out)
- Position (x, y coordinates)
- Live/dead status relative to striker's ball

Track for turn:
- Which balls have been roqueted this turn
- Whether a hoop has been run since last roquet

### 3. Shot Physics
Croquet strokes move two balls. Key ratios:
- **Stop shot**: Striker goes 1/5 to 1/10 the distance of croqueted ball
- **Drive**: Striker goes 1/3 the distance (standard stroke)
- **Half roll**: Striker goes 1/2 the distance
- **Full roll**: Both balls travel equal distance
- **Pass roll**: Striker goes FURTHER than croqueted ball
- **Split**: Balls diverge (max 90° apart)
- **Takeoff**: Croqueted ball barely moves, striker travels

### 4. AI Decision Making
Use the tactical content to inform AI choices:

**Opening (turns 1-4)**:
- First ball: East boundary between H4-H5 level
- Second ball: Standard tice (8-13 yards from corner) or defensive corner
- Third/fourth: Shoot at balls or establish break

**During breaks**:
- Prioritize 4-ball break (easiest to maintain)
- Place pioneers at next hoop
- Keep pivot ball mid-court
- Run hoops with control to get good rushes

**Leaves (end of turn)**:
- Diagonal Spread Leave (DSL): Most common, forgiving
- New Standard Leave (NSL): More sophisticated
- Goal: No short shots for opponent, good pickup for partner

**Risk assessment**:
- Distance affects hit probability
- Cut rushes harder than straight rushes
- Consider opponent's likely response

### 5. Difficulty Factors
Shot success probability should consider:
- Distance to target
- Angle (cuts are harder)
- Hoop approach angle
- Whether stroke is hampered
- Player skill level (for handicap play)

## DO NOT Read

The PDF files are too large - they will cause errors:
- `pdfs/AC-Laws-Rulings-Commentary-Combined.pdf`
- `pdfs/croquetcoachinghandbook.pdf`
- `pdfs/WCF-GC-Rules-6th-Edition-Final-7.3.22.pdf`

Use the `.txt` summaries instead.

## Example: Implementing a 4-Ball Break

From `basic-ac-tactics.txt`:

```
1. Roquet the reception ball (near current hoop)
2. Croquet stroke: Send it toward NEXT hoop as pioneer
3. Run current hoop with control
4. Rush reception ball toward pivot
5. Roquet pivot ball
6. Croquet stroke: Send pivot to good position, go to pioneer
7. Roquet pioneer (now at next hoop)
8. Croquet stroke: Get position for hoop approach
9. Repeat from step 3
```

The key insight: "Having all four balls available means no difficult strokes need ever be played."

## Questions to Answer from Docs

When implementing features, consult the docs for:
- "What's a legal move here?" → `ac-laws-summary.txt`
- "What should the AI do?" → `basic-ac-tactics.txt`, `wylie/`
- "How hard is this shot?" → `coaching/`, shot ratios in tactics
- "What leave should AI make?" → `basic-ac-tactics.txt` (Leaves section)
- "What's the opening strategy?" → `basic-ac-tactics.txt` (Openings section)

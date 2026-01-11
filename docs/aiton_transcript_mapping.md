# Aiton Teachings Cross-Referenced with Elite Game Transcripts

## Overview

This document maps Keith Aiton's tactical teachings to concrete examples from elite tournament play (2022-2025), validating reward values and identifying gaps.

---

## 1. Hoop Approaches (Aiton Section 2.3)

### Aiton Teaching:
- Ideal approach: **1 yard in front** with stop-shot ratio ~1:6
- Right side approaches easier than left
- 12 inches requires excellent stop-shot control
- Angle significantly affects difficulty

### Transcript Validation:

| Transcript Example | Aiton Concept | Outcome |
|-------------------|---------------|---------|
| "Bamford misapproaches 2-back from close range" (2022 Final G1T4) | Close approach requires excellent control | Failed - confirms difficulty |
| "runs slightly angled hoop" (2023 Final G5T3) | Angle affects difficulty | Success with "nice rush" setup |
| "makes crucial three-foot hoop with rush" (2023 Final G1T5) | ~1 yard ideal distance | Success |
| "Death blue taps yellow, rushes red to black" (2023 Final G1T5) | Rush to approach pattern | Break continuation |

### Reward Calibration:
- **Aiton code**: `IDEAL_APPROACH_DISTANCE = 1.0`, `MAX_GOOD_APPROACH = 3.0`
- **Transcript evidence**: Misapproaches happen even at close range; confirms approach isn't trivial
- **Suggested reward**: Keep current; add penalty for rushed/angled approaches

---

## 2. Leave Types (Aiton Section 2.5)

### Aiton Teaching:
- **NSL (New Standard Leave)**: Partner at hoop 2, opponents separated
- **Diagonal Spread**: Balls to corners for defense
- **MSL (Maugham Standard Leave)**: Variation on NSL

### Transcript Validation:

| Transcript Example | Aiton Concept | Context |
|-------------------|---------------|---------|
| "Mark completes three-ball break into **New Standard Leave** with rush to hoop 1" (2023 Final G5T6) | NSL | Used before opponent's turn - defensive |
| "diagonal spread with red at peg" (2023 Semis G2T4) | Diagonal Spread variant | Mid-game defensive positioning |
| "leaves 'three ducks' position" (2023 Semis G4T3) | Named leave pattern | Advanced tactical leave |

### Gap Identified:
- **"Three ducks"** leave not in Aiton code - should add
- NSL confirmed as elite-level standard

### Reward Calibration:
- **Current**: `leave_setting` avg_reward=0.17 (18 instances)
- **Evidence**: NSL directly leads to opponent miss + break pickup
- **Suggested**: Increase leave_setting reward to 0.5-0.8 for proper leaves

---

## 3. Break Building (Aiton Sections 2.4-2.6)

### Aiton Teaching:
- **Pioneer**: 3-4 yards in front of NEXT hoop
- Reception ball positioning determines approach quality
- 3-ball to 4-ball transition is critical

### Transcript Validation:

| Transcript Example | Aiton Concept | Evidence |
|-------------------|---------------|----------|
| "rushes blue 1y N of 1, makes hoop with break" (2023 Final G1T3) | Reception ball near hoop | 1 yard = ideal approach |
| "picks up break" (2023 Final G2T5, G3T4) | Break establishment | Commonly referenced |
| "three-ball break into NSL" (2023 Final G5T6) | Break continuation | 3-ball break is standard |
| "establishes 4-ball break" (2022 Semis multiple) | 4-ball break | Gold standard |

### Key Insight:
- Pioneer placement is **implicit** in transcripts ("makes hoop with break" implies pioneers in place)
- Aiton explicitly teaches it; commentators assume it
- This explains why `pioneer_placement` has only 4 explicit mentions

### Reward Calibration:
- **Current**: `pioneer_placement` avg_reward=0.60 (4 instances)
- **Evidence**: Every successful break implies proper pioneers
- **Suggested**: Increase to 2.0-3.0; it's prerequisite for all breaks

---

## 4. Peel Sequences

### Aiton Teaching:
- Not covered in "The Basics" chapter (advanced topic)

### Transcript Evidence (Teaching Extension):

| Peel Type | Count | Example |
|-----------|-------|---------|
| Triple Peel (TP) | 95 | "Death wins +26tp" - standard finish |
| Sextuple (SXP) | 8 | "Bamford executes all six peels before 3-back" |
| Quintuple (QP) | 4 | "converts to quintuple" |
| OTP (on opponent) | 3 | "Avery wins +14otp" |

### Peel Timing Pattern (from transcripts):
```
"peels black through 4-back after 3, through penult after 6, through rover before 3-back"
```
This reveals standard TP timing:
- 4-back peel: after running hoop 3
- Penult peel: after running hoop 6
- Rover peel: before running 3-back

### Reward Calibration:
- **Current**: `peel_sequence` avg_reward=1.76 (95 instances)
- **Evidence**: TPs are the standard winning finish
- **Suggested**: Keep current; add bonus for completed TP (+5.0)

---

## 5. Supershot Opening (NOT in Aiton)

### Gap Identified:
Aiton's "Basics" chapter doesn't cover opening theory, but transcripts show a clear pattern:

| Position | Frequency | Example |
|----------|-----------|---------|
| "2y NNW of 5" | 3 | 2023 Final G1T1, G2T1, G3T1 |
| "3y NNW of 5" | 2 | 2023 Final G4T1 |
| "4y SSW of peg" | 2 | 2022 Semis |
| "10y N of IV" | 3 | 2023 Semis |

### New Concept to Add:
```python
# Supershot opening positions (from transcripts)
SUPERSHOT_POSITIONS = [
    Vector2(12, 22),  # 2-3y NNW of hoop 5
    Vector2(14, 14),  # Near peg, SSW
    Vector2(25, 10),  # 10y N of corner IV
]
```

---

## 6. Position vs Shooting (Expert Tactics)

### Aiton-Maugham Teaching:
- "What do you do after you've run this?"
- Wired positions valuable against poor shooters

### Transcript Validation:

| Transcript Example | Concept |
|-------------------|---------|
| "Avery yellow misses blue from A-baulk" (2023 Final G1T4) | Long shot from baulk often misses |
| "Death blue misses target from B-baulk" (2023 Final G2T4) | Baulk shots risky |
| "Mark makes double from south boundary, misses" (2023 Final G5T4) | Double targets very difficult |
| "James lifts black, misses long shot to corner 4" (2023 Final G5T6) | Long shots frequently miss |

### Reward Calibration:
- **Transcript evidence**: Shots from baulk/boundary miss frequently
- **Expert code**: `shot_confidence = 0.30` for >12 yards
- **Validation**: Correct - elite players miss 12+ yard shots regularly

---

## 7. Terminology Mapping

### Aiton Code Term → Transcript Notation

| Aiton Code | Transcript Notation | Example |
|------------|---------------------|---------|
| `LeaveType.NSL` | "New Standard Leave" | G5T6 |
| `LeaveType.DIAGONAL_SPREAD` | "diagonal spread" | Semis G2T4 |
| `IDEAL_APPROACH_DISTANCE` | "1y N of 1" | G1T3 |
| `IDEAL_PIONEER_DISTANCE` | (implicit in "break") | Throughout |
| `ApproachSide.RIGHT/LEFT` | "slightly angled" | G5T3 |

### New Terms from Transcripts (not in code):

| Term | Meaning | Add to Code? |
|------|---------|--------------|
| "supershot" | Opening ball placement near hoop 5 | Yes |
| "three ducks" | Specific leave pattern | Yes |
| "fifth turn finish" | Win by turn 5 | Metric only |
| "OTP" | Triple peel on opponent's ball | Yes |
| "rush-peel" | Peel executed via rush | Yes |

---

## 8. Reward Value Recommendations

Based on transcript frequency and outcome correlation:

| Pattern | Current Reward | Suggested | Rationale |
|---------|---------------|-----------|-----------|
| pioneer_placement | 0.60 | **2.5** | Prerequisite for all breaks |
| leave_setting | 0.17 | **0.8** | NSL leads to opponent miss |
| break_assembly | 1.58 | 1.6 | Correct |
| peel_sequence | 1.76 | 1.8 | Correct |
| break_continuation | 2.26 | 2.3 | Correct |
| rush_control | 1.82 | 1.8 | Correct |
| supershot_opening | N/A | **1.0** | Add new pattern |
| completed_tp | N/A | **5.0** | Add bonus for TP finish |

---

## 9. Implementation Priorities

### High Priority (Missing from current system):
1. Add supershot opening recognition
2. Increase pioneer placement reward (currently severely undervalued)
3. Add "three ducks" and other named leaves
4. Add OTP (peel on opponent) pattern

### Medium Priority (Calibration):
1. Adjust leave_setting reward upward
2. Add completed peel sequence bonuses
3. Track peel timing (after hoop 3, after hoop 6, etc.)

### Low Priority (Edge cases):
1. Rush-peel technique recognition
2. Self peg-out tactical detection
3. Extended defensive exchange patterns

---

## 10. Key Insight

**The transcripts reveal that Aiton's teachings are correct but commentators assume the knowledge.**

Pioneer placement appears only 4 times explicitly, but every "picks up break" and "makes hoop with break" implies proper pioneers. The reward system should treat pioneer placement as a **prerequisite** with high reward, not an optional pattern.

This validates Aiton while showing why naive transcript parsing undervalues fundamental tactics.

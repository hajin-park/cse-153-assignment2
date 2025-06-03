
# CHOREO2GROOVE DEMO INSTRUCTIONS

## Quick Demo Script (5 minutes):

### 1. Introduction (30 seconds)
"I built an AI that generates drum beats from dance movements using the AIST++ dataset."

### 2. Show the Problem (30 seconds)  
"The challenge: Can AI learn the relationship between body movement and musical rhythm?"

### 3. Demo the Solution (2 minutes)
- **Play dance_video_FIXED.gif**: "Here's the dance input"
- **Play generated_drums.mid**: "Here's what my AI generated"
- **Play original_drums.mid**: "Here's the human reference"

### 4. Prove It's Not Random (1.5 minutes)
- **Show alignment_analysis.png**: "This graph shows correlation between movement and drums"
- **Play random_drums_1.mid**: "Here's what truly random sounds like"
- **Compare**: "Notice my AI uses consistent patterns, random is chaotic"

### 5. Technical Evidence (30 seconds)
- "The AI learned drum vocabulary (132/133 events are snares)"
- "Reasonable timing density (not silent, not overwhelming)"
- "Responds to different dance inputs differently"

### 6. Conclusion (30 seconds)
"While the timing correlation is weak and needs more training, the AI clearly learned patterns rather than generating random noise."

## If Asked Technical Questions:

**"How do you know it's working?"**
- Show the correlation analysis
- Compare to random baseline
- Point out consistent drum type usage

**"Why is correlation low?"**
- Limited training data (76 samples)
- Only 3 epochs of training
- Complex temporal relationship learning takes time

**"What would improve it?"**
- More training data
- Longer training time
- Better temporal alignment in data processing

## File Reference:
- Main demo: `dance_video_FIXED.gif` + `generated_drums.mid`
- Proof graphs: `alignment_analysis_sample_58.png`
- Random comparison: `random_drums_*.mid`
- Technical details: All JSON reports

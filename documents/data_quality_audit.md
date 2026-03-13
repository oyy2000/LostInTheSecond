# Training Data Quality Audit

**Dataset**: `samples_gsm8k_train_ds2_fix_step2_gpt_prefill.json`
**Total samples**: 115
**Date**: 2026-03-12 15:02

## Summary

| Metric | Count | Percentage |
|--------|-------|------------|
| Clean (no issues) | 60 | 52.2% |
| Flagged (has issues) | 55 | 47.8% |
| pos_response gives wrong answer | 0 | 0.0% |
| neg_response gives correct answer | 15 | 13.0% |
| Missing/short corrected_step2 | 0 | 0.0% |

## Issue Type Breakdown

| Issue Type | Count |
|------------|-------|
| Prefill change is unexpected: unchanged_wrong | 48 |
| neg_response is actually CORRECT | 15 |
| Step count mismatch: pos=13, neg=7 | 2 |
| Step count mismatch: pos=6, neg=13 | 1 |
| Step count mismatch: pos=20, neg=13 | 1 |
| Step count mismatch: pos=15, neg=7 | 1 |
| Step count mismatch: pos=6, neg=12 | 1 |
| Step count mismatch: pos=19, neg=7 | 1 |
| Step count mismatch: pos=26, neg=13 | 1 |
| Step count mismatch: pos=8, neg=17 | 1 |
| Step count mismatch: pos=3, neg=11 | 1 |
| Step count mismatch: pos=7, neg=17 | 1 |
| Step count mismatch: pos=5, neg=12 | 1 |
| Step count mismatch: pos=14, neg=8 | 1 |
| Step count mismatch: pos=3, neg=13 | 1 |
| Step count mismatch: pos=18, neg=11 | 1 |
| Step count mismatch: pos=18, neg=9 | 1 |
| Step count mismatch: pos=17, neg=11 | 1 |

## Flagged Samples (Require Manual Review)

| Idx | QID | Question (truncated) | Issues |
|-----|-----|---------------------|--------|
| 43 | 2340 | Jamal works at a library shelving books. He has a cart full of books to put away | neg_response is actually CORRECT (59 == 59); Step count mismatch: pos=3, neg=11; Prefill change is unexpected: unchanged_wrong |
| 64 | 4327 | Samuel is going to the cinema with his brother, Kevin. They both have a total bu | neg_response is actually CORRECT (0 == 0); Step count mismatch: pos=5, neg=12; Prefill change is unexpected: unchanged_wrong |
| 70 | 4979 | Annie spends 2 hours a week on chess club, 8 hours a week on drama club, and 3 h | neg_response is actually CORRECT (130 == 130); Step count mismatch: pos=13, neg=7; Prefill change is unexpected: unchanged_wrong |
| 77 | 5391 | Reggie is playing marbles with his friend. His friend arrives with 100 marbles.  | neg_response is actually CORRECT (0 == 0); Step count mismatch: pos=18, neg=11; Prefill change is unexpected: unchanged_wrong |
| 3 | 322 | A shady restaurant is charging customers gratuities after taxes without them bei | Step count mismatch: pos=6, neg=13; Prefill change is unexpected: unchanged_wrong |
| 4 | 404 | A hay farmer harvested 560 bales of hay from 5 acres of grass per month last yea | Step count mismatch: pos=20, neg=13; Prefill change is unexpected: unchanged_wrong |
| 9 | 868 | A flagpole is 12 feet tall. It breaks, folding over in half, such that what was  | Step count mismatch: pos=19, neg=7; Prefill change is unexpected: unchanged_wrong |
| 11 | 975 | John hits 70% of his free throws.  For every foul he gets 2 shots.  He gets foul | neg_response is actually CORRECT (160 == 160); Prefill change is unexpected: unchanged_wrong |
| 14 | 1081 | John has 2 hives of bees.  One of the hives has 1000 bees and produces 500 liter | neg_response is actually CORRECT (560500 == 560500); Prefill change is unexpected: unchanged_wrong |
| 20 | 1363 | Prudence was starting a cupcake business.  She figured that each cupcake cost $0 | neg_response is actually CORRECT (72 == 72); Prefill change is unexpected: unchanged_wrong |
| 24 | 1504 | A small theater company sells tickets to a show.  They have a 400 seat theater a | neg_response is actually CORRECT (28800 == 28800); Prefill change is unexpected: unchanged_wrong |
| 31 | 1777 | A school is getting ready to open for the year and the sports class is organizin | neg_response is actually CORRECT (60 == 60); Prefill change is unexpected: unchanged_wrong |
| 32 | 1828 | Joseph invested $1000 into a hedge fund. The fund promised a yearly interest rat | Step count mismatch: pos=26, neg=13; Prefill change is unexpected: unchanged_wrong |
| 36 | 1931 | At a shop in Japan, women's T-shirts are sold every 30 minutes for $18, and men' | Step count mismatch: pos=8, neg=17; Prefill change is unexpected: unchanged_wrong |
| 52 | 2937 | The lunchroom is full of students: 40% are girls and the remainder are boys. The | neg_response is actually CORRECT (84 == 84); Prefill change is unexpected: unchanged_wrong |
| 58 | 3466 | UF got into the national championship. For them to get into the championship, th | neg_response is actually CORRECT (15 == 15); Prefill change is unexpected: unchanged_wrong |
| 60 | 3645 | Geordie takes a toll road on his drive to work and back every day of his five-da | neg_response is actually CORRECT (66.50 == 66.50); Prefill change is unexpected: unchanged_wrong |
| 62 | 3930 | Jonsey is awake for 2/3 of the day and spends 1/2 her time awake playing outside | neg_response is actually CORRECT (\frac{5 == \frac{5); Prefill change is unexpected: unchanged_wrong |
| 88 | 5920 | In a store, there are three types of cheese: white, swiss, and blue cheese. Each | neg_response is actually CORRECT (69 == 69); Prefill change is unexpected: unchanged_wrong |
| 89 | 6010 | James can do a farmer's walk with 300 pounds per hand for 20 meters.  He can lif | Step count mismatch: pos=13, neg=7; Prefill change is unexpected: unchanged_wrong |
| 99 | 6832 | The time Juan takes to grab his lunch from his office and back is half the time  | neg_response is actually CORRECT (500 == 500); Prefill change is unexpected: unchanged_wrong |
| 5 | 405 | A river is to be used for a boat race. If each boat is 3 feet across and they mu | Step count mismatch: pos=15, neg=7 |
| 6 | 486 | Emily wants to know how much it rained last week. She sees that it rained 2 inch | Step count mismatch: pos=6, neg=12 |
| 8 | 803 | An elementary school teacher is making Halloween goodie bags for her class.  She | Prefill change is unexpected: unchanged_wrong |
| 10 | 951 | Michonne is inviting her friends to her birthday party. She invites 6 of her fri | Prefill change is unexpected: unchanged_wrong |
| 19 | 1246 | A spaceship is traveling to another planet. The spaceship travels at a consisten | Prefill change is unexpected: unchanged_wrong |
| 21 | 1386 | An auto shop buys tires to replace all the tires on every customer’s car. They b | Prefill change is unexpected: unchanged_wrong |
| 25 | 1540 | Sam invested $10,000 and earned 20% interest compounded for 3 years. He then inv | Prefill change is unexpected: unchanged_wrong |
| 28 | 1651 | John builds a box.  The box is 26 inches by 26 inches by 14 inches.  The walls a | Prefill change is unexpected: unchanged_wrong |
| 29 | 1692 | Sue is traveling from New York to San Francisco, 16 hours later after landing in | Prefill change is unexpected: unchanged_wrong |
| 33 | 1842 | Carmen is counting the cars that pass by her window. All the cars are either whi | Prefill change is unexpected: unchanged_wrong |
| 34 | 1868 | Alex and Max are running a race against each other. At the beginning of the race | Prefill change is unexpected: unchanged_wrong |
| 42 | 2337 | The P.T.O. decided to provide shirts for the elementary students for track and f | Prefill change is unexpected: unchanged_wrong |
| 45 | 2494 | Sydney and Conner are having a three day rock collecting contest to see who can  | Prefill change is unexpected: unchanged_wrong |
| 48 | 2605 | John adopts a dog from a shelter.  The dog ends up having health problems and th | Prefill change is unexpected: unchanged_wrong |
| 51 | 2750 | Zoe made a total of $8,000 cleaning pools and babysitting. She babysat Julie thr | Step count mismatch: pos=7, neg=17 |
| 55 | 3145 | It will cost $60 to rent a sailboat and $80 per hour to rent a ski boat. Ken ren | Prefill change is unexpected: unchanged_wrong |
| 67 | 4666 | Dana has 15 more pencils than Jayden, who has twice as much as Marcus. How many  | Step count mismatch: pos=14, neg=8 |
| 69 | 4963 | A TV show costs $100,000 per episode for the first season and twice that much fo | Prefill change is unexpected: unchanged_wrong |
| 71 | 5015 | Jack cycles from his home to the store. Then he cycles, at the same speed, 50 mi | Prefill change is unexpected: unchanged_wrong |
| 75 | 5289 | John decides to stop delivering the newspapers he is supposed to deliver and ins | Prefill change is unexpected: unchanged_wrong |
| 76 | 5369 | Matt has a peanut plantation that is 500 feet by 500 feet.  1 square foot of pea | Step count mismatch: pos=3, neg=13 |
| 79 | 5440 | Each week Jaime saves $50. Every two weeks she spends $46 of her savings on a ni | Prefill change is unexpected: unchanged_wrong |
| 80 | 5521 | It takes 1 hour for refrigerated dough to come to room temperature.  It takes 15 | Prefill change is unexpected: unchanged_wrong |
| 85 | 5861 | Ralph has $54.00 worth of products in his cart.  At the register, he asks if he  | Prefill change is unexpected: unchanged_wrong |
| 90 | 6044 | Tom plays 9 rounds of golf.  He takes an average of 4 strokes per hole.  The par | Prefill change is unexpected: unchanged_wrong |
| 96 | 6575 | Bob is building raised beds for his vegetable garden. Each bed is 2 feet high, 2 | Prefill change is unexpected: unchanged_wrong |
| 97 | 6641 | A giant spider is discovered.  It weighs 2.5 times the previous largest spider,  | Step count mismatch: pos=18, neg=9 |
| 98 | 6714 | Paul's grades last semester were very bad. To encourage him, Paul's dad promised | Prefill change is unexpected: unchanged_wrong |
| 103 | 6914 | Billy wants to watch something fun on YouTube but doesn't know what to watch.  H | Prefill change is unexpected: unchanged_wrong |
| 105 | 7039 | Jerry can run from his house to his school and back in the time it takes his bro | Step count mismatch: pos=17, neg=11 |
| 108 | 7335 | Sophie does 4 loads of laundry a week and uses 1 dryer sheet per load.  A box of | Prefill change is unexpected: unchanged_wrong |
| 109 | 7358 | Jack has been driving for the past 9 years. He drives 37,000 miles every four mo | Prefill change is unexpected: unchanged_wrong |
| 110 | 7364 | Angelo and Melanie want to plan how many hours over the next week they should st | Prefill change is unexpected: unchanged_wrong |
| 111 | 7370 | Jame gets 20 singing lessons.  He gets the first lesson free and after the first | Prefill change is unexpected: unchanged_wrong |

## Per-Sample Statistics

| Metric | Value |
|--------|-------|
| Mean pos_steps | 8.7 |
| Mean neg_steps | 8.6 |
| Mean corrected_step2 length | 288 chars |

## Prefill Correctness Changes

| Change Type | Count |
|-------------|-------|
| wrong_to_correct | 67 |
| unchanged_wrong | 48 |

---
*Generated: 2026-03-12 15:02*

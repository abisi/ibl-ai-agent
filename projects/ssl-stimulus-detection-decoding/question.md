# SSL stimulus-detection decoding

## Request (Axel Bisi, 2026-10-16)
Repeat the analyses of ssl-stimulus-arrival-decoding (whisker vs auditory, i.e. stimulus identity) but decoding stimulus
presence: whisker vs correct rejection and auditory vs correct rejection. Correct rejections (CR) = no_stim_trial with
lick_flag 0 (plentiful). Whisker trials outnumber auditory trials ("that's how it is"). As in the arrival project, task
(active) and passive trials, decoded separately and compared. One project for both targets.
Next step (after the decoding): relate decoding strength / onset per area to anatomy, as a sanity check: long-range axonal
projections of wS1 and wS2 (Liu et al. 2024, eLife) and Allen Mouse Connectivity injections in whisker or auditory cortices.

## Method (decided with the user, 2026-10-16)
| Item | Decision |
|---|---|
| Pipeline | ssl-stimulus-arrival-decoding unchanged (decoder, sampling, eligibility, bins, areas, onset rule, N list, iterations) except the rows below |
| Targets | whisker vs no stimulus; auditory vs no stimulus (DETECT_TARGET) |
| Task epoch | no stimulus = correct rejections (no_stim_trial, lick_flag 0); same trial rules as arrival (perf != 6, warm-up cut, A1 trim); stimulus trials lick or not |
| Passive epoch | passive blocks have (almost) no no-stimulus trials (0-1 per session), so no stimulus = the window just before a passive stimulus (user): sham onset 0.8 s before the stimulus of the OTHER modality's passive trials (whisker target: before auditory stimuli, and vice versa), so each trial gives one event and the labels follow the trial sequence; the sham's full window (-200..+600 ms) ends 200 ms before the stimulus; lick-free shams only |
| Null | linear shift only (user): labels of the time-ordered events shifted by 10-50 % of the events, non-wrapping, random direction (ssl-pseudopopulation-area-decoding 002 make_parts); no shuffle null. Stimulus and CR trials are spread differently through a session, so a shuffle null would let slow drift count as information |
| Artefact | Poisson replacement of -10..+5 ms at every whisker onset and at every event onset of every class (CR and sham included), so that window cannot separate the classes |
| Class imbalance | handled by the balanced pseudo-trials (100 per class); only eligibility needs >= 3 events per class |
| Iterations | N sweep 100 x 10 shifts (pilot, as arrival); final N = 200: 500 x 20 |

## Notes
- With two events per passive trial (its pre-window and its stimulus), labels would strictly alternate and a linear shift
  could only re-align or invert them; hence the other-modality pre-window (user choice, 2026-10-16).

# Research direction update: 2 November 2026

Run this in my research project folder, which holds my paper code and documents. Treat everything below as decisions and source facts. Do not add assumptions.

## Decisions
1. My master's thesis is based on Paper 1 and Paper 2.
2. All existing Paper 3 material, including my deep searches and evaluations, now belongs to Paper 4.
3. Paper 3 now has two options: A3 Haratres4t and A4 B2B Matchmaking. Only one will be chosen. I have not chosen yet.
4. I have been considering LLM or agent-based work for Paper 3. A3 and A4 both use LLMs. Neither slide mentions agents, so do not describe either option as agent-based unless I confirm it.
5. Per my last chat with Prof. Ahmet Sayar, he also suggested B2B Matchmaking.
6. From now on, I want to track PhD and research positions carefully so I can apply on time.

## Paper 3 options, from Prof. Sayar's 2 Nov 2026 slides
Both are listed under Category A: TÜBİTAK-Funded Projects.

**A3, Project 03 of 09: Haratres4t, Explainable AI-Based Product Quality Scoring**
- Problem: Single average star ratings in e-commerce are easily manipulated by fake reviews, fail to highlight feature-level trade-offs (e.g. durability vs delivery), and penalize new products with high statistical variance.
- Method: Processes text, ratings, and user signals using a multilingual LLM for aspect-based sentiment analysis. Evaluates review credibility and models quality via a hierarchical Bayesian framework with epistemic uncertainty adjustments.
- Contribution: Establishes a unified quality scoring policy integrating aspect sentiment, review credibility, and sample-size uncertainty calibration to prevent artificial rating distortion.
- Use case: E-commerce decision portal presenting shoppers with confidence intervals and aspect breakdowns (e.g. sound quality vs battery life) to guide purchases.

**A4, Project 04 of 09: GraphRAG & LLM-Based Contextual B2B Matchmaking**
- Problem: Traditional B2B matching relies on keyword or rule-based heuristics, missing semantic domain terminology, multi-hop supply chain synergies, and historical interaction outcomes, which reduces event ROI for SMEs.
- Method: Integrates company profiles, adapted Turkish LLMs, and knowledge graphs using GraphRAG multi-hop retrieval to compute contextually aligned B2B recommendations with natural language rationales.
- Contribution: Presents an explainable, Turkish-aware B2B matchmaking framework capturing multi-hop business relationships, validated on real-world event data from Oniki.
- Use case: SME manufacturing firms finding international industrial automation distributors through structured multi-hop capability and market alignment explanations.

The "Project 03" and "Project 04" labels are slide numbers, not my paper numbers.

## What Prof. Sayar said in his 2 Nov 2026 voice notes
- He considers my master's thesis complete and says I have a publication from it.
- I should ask the institute whether I can graduate early. If I must complete two semesters, he suggests graduating at the start of the spring term, around January–February.
- He also said graduating with good publications matters more than just finishing. These two points have not been reconciled.
- For a PhD scholarship request he wants to be able to cite 3–4 publications, so our publications need to be strong.
- He talked about writing a third and fourth paper together.
- If I continue to a PhD, I can take up these project topics there.
- He said he would try to arrange a PhD scholarship at the university and find TÜBİTAK project funding. Record these as efforts he offered, not as confirmed.

## Your task
### Phase 1: read-only audit, then stop
Change nothing in this phase.
1. Map where the code, data, manuscripts, results and submission files for each paper live.
2. Assess Paper 2's readiness for submission. List exactly what remains, with file references. Mark anything you cannot verify.
3. Check whether each number in Paper 2's manuscript can be traced to code or results in this project. List any that cannot.
4. List every file, folder, note, link and reference that belongs to the current Paper 3, and show how each would move to Paper 4.
5. Find any existing material on thesis writing, A3, A4, or positions and applications.
6. Report which documents are out of date or contradict the decisions above.
7. Propose the exact changes for Phase 2, then wait for my approval.

### Phase 2: after I approve
1. Update the project's main instructions or status file with the decisions above.
2. Relabel all existing Paper 3 material as Paper 4, keeping its content intact and updating references to it. Preserve its history if the project uses git.
3. In the Paper 3 location, add A3 and A4 as the two options, using the slide text above. Do not create a separate Paper 3/4 workspace. Leave anything the slides do not establish, including data access and scope, as an open question.
4. Add a thesis workspace that maps thesis content to Paper 1 and Paper 2, with my open questions listed.
5. Add a Paper 2 submission checklist based on the audit.
6. Add a positions tracker, as described below.
7. Create a short next-actions list, ordered by what is blocking what.

## Positions tracker
- Start with these sources: NTNU/NorwAI, KTH, UiO, Chalmers, and the scholarship and TÜBİTAK routes Prof. Sayar mentioned.
- Search official university, group and funder pages. Add only openings you can verify, with a link.
- Record for each one: institution, group or supervisor, position title, topic, link to the official posting, deadline with time zone, requirements, documents needed, my status, date last checked, and next action.
- Sort open positions by nearest deadline. Flag any deadline within 30 days.
- If a deadline or requirement is unclear, write "not confirmed". Do not guess.
- Keep sources with no current opening on a watchlist, with the date last checked. Do not list them as open positions.
- Do not rate my fit unless I ask.
- Never apply, register, or contact anyone on my behalf.

## Open questions: do not resolve these yourself
- Whether A3 or A4 becomes Paper 3.
- What happens to the option I do not choose.
- What the institute says about early graduation, and how that fits with building publications first.
- Paper 2's target venue, unless it is already recorded in the project.

## Rules
- Do not invent or infer results, citations, venues, dates, datasets, data access or position details.
- Do not change results, data, or the wording of the published Paper 1.
- Follow the project's existing structure and naming. Prefer updating existing files over creating duplicates.
- Do not delete files. Move files only as approved in Phase 2.
- Do not commit, push or submit anything unless I ask.
- Keep source facts, decisions and your own suggestions clearly separate, and label suggestions as suggestions.

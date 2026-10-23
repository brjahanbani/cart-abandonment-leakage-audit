# Paper 2 · Data and ethics statement (draft)

Draft text for the paper's data-availability and ethics section. Items marked **[CONFIRM]**
need the author's confirmation before submission.

## Retail Rocket
Public dataset (Kaggle, "Retailrocket recommender system dataset"), as used and cited in paper 1.

## REES46 cosmetics shop
Public dataset: Kaggle, "eCommerce Events History in Cosmetics Shop" (owner Michael Kechinov),
listed under "Data files © Original Authors". The page states: *"You can use this dataset for
free. Just mention the source of it: link to this page and link to REES46 Marketing Platform."*
→ Cite the Kaggle page and https://rees46.com, and thank REES46 in the acknowledgements.
(Checked 2026-10-20.)

## Flower-shop dataset (private)

Draft wording:

> The second dataset comes from the production event log of an online flower shop, which is
> kept anonymous. The author develops the shop's web platform and designed its event
> collection; the data are used with the written permission of the shop's owner. All
> analyses use pseudonymous browser
> and customer identifiers. Personal fields (delivery-driver names and phone numbers, payment
> references, customers' gift notes) were removed on load and never analysed, and no raw
> records leave the author's machine. Because the data contain customers' purchase histories,
> they cannot be shared. The code and all aggregate results are archived on Zenodo
> (https://doi.org/10.5281/zenodo.23601825) and maintained at
> https://github.com/brjahanbani/cart-abandonment-leakage-audit. The site's
> terms of use include a privacy commitment to protect users' personal information and
> reserve the use of site data to the shop, subject to its written permission; the research
> use relies on that permission and is limited to pseudonymised, aggregate analysis.

### Points to confirm
1. ~~Name the shop, or keep it anonymous?~~ **Anonymous** (author, 2026-10-21). No shop name, city, domain or database name in the paper or the public repository.
2. ~~Your role?~~ **Developer of the shop's website** (author, 2026-10-21). So the **owner's written permission is needed** (an email is enough), covering: use of the pseudonymised event and order logs for research, publication of aggregate results, and that the shop stays unnamed.
3. ~~Privacy policy or terms?~~ **Reviewed 2026-10-21** (terms provided by the author): a general privacy commitment; no mention of analytics, usage data or research; site data reserved to the shop, subject to its written permission. The owner's written permission is therefore the basis. **Received by email on 2026-10-23** (kept by the author), in the agreed wording: use of the anonymised event and order records for academic research and publication of aggregate results, on condition that the shop is not named and no personal customer data are published. The author also promised to send the owner the paper before submission.
4. ~~Repository?~~ **Changed 2026-10-23:** the earlier repository, whose history identified the shop, is now private. Code and aggregate results are public in a clean repository (`brjahanbani/cart-abandonment-leakage-audit`, no shop history) archived on Zenodo (DOI 10.5281/zenodo.23601825, release v1.0). The shop's own website code is not involved.
5. Ethics approval: **asked Prof. Sayar** (email of 2026-10-21). **[PENDING]**

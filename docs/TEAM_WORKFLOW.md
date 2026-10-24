# Team T042 workflow

## Everyone, once (two commands)
```bash
git clone https://github.com/Pranav-Singh-Devloper/traffic-oracle.git
cd traffic-oracle && python3 join.py
```
That is all. `join.py` identifies you (GitHub login or git email), sets your git name and email, builds the environment, downloads the dataset, runs the checks for **your** modules, records the real result in `docs/viva/<you>.md`, commits as **you**, pushes your branch `<you>/first-verification` and opens the pull request. It stops with a clear message if anything fails, and is safe to run again.

Needs only `git` and Python 3. Optional: the GitHub CLI (`gh auth login`) makes it open the pull request for you; without it, it prints the link.
Your git email must be verified on your GitHub account (GitHub -> Settings -> Emails), otherwise GitHub will not credit the commit to you.

| Member | GitHub | Email registered for commits |
|---|---|---|
| Pranav Singh | Pranav-Singh-Devloper | raindeer.social@gmail.com / pranavsinghprimo@gmail.com |
| Kunal | kunaldevsahu | kunal.sahu2024@nst.rishihood.edu.in |
| Himanshu | HimanshuPal29 | himanshu.pal2024@nst.rishihood.edu.in |
| Kanchan | kanchan97-cp | bansalkanchan23@gmail.com |
| Vaibhav | VaibhavSingh50 | vaibhavsingh0564@gmail.com |

## Pranav
Review and merge the pull requests (`.github/CODEOWNERS` requests reviews automatically).

## Later (optional)
- `python scripts/owners.py mine --as <you>` lists your files; `python scripts/owners.py commit --as <you> "message"` commits only your changed files as you (it checks your git email); `python scripts/owners.py stats` shows commits per author.
- Improve your own files (theory notes, tests, viva answers in your own words) on a branch `<you>/<topic>` and open a pull request.

## Honest authorship
Most of the initial code and docs were written with an AI assistant and committed from Pranav's account as a baseline. Do not claim otherwise in the report or viva; check your course's rules on AI-assisted work and disclose accordingly.

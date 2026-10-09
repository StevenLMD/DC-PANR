# Release checklist / 开源步骤

## Before making the repository public

1. Verify copyright and software-release permission with **all relevant coauthors, employers and institutions**; confirm any patent/technology-transfer, confidentiality, grant, and third-party-source constraints.
2. Review `LICENSE` (MIT) and change it **before publication** if the legitimate rights holders require a different authorized license. The package assumes the rights holders approve MIT; it cannot grant rights on their behalf.
3. Confirm the paper authors, DOI and metadata in `CITATION.cff`.
4. Ensure `outputs/`, `.venv/`, `*.pt`, private datasets and researcher credentials are excluded. This ZIP deliberately contains no pretrained weights, caches, experiments or private observations.
5. Locally execute `python -m unittest discover -s tests -v` and `python run_once.py --quick`; never cite smoke performance as peer-reviewed experimental results.

## Publish on GitHub (web interface)

1. Open <https://github.com/new>, choose owner and repository name (suggested: `DC-PANR`) and visibility `Public`.
2. Create the repository without an auto-generated README/license/gitignore (they are already supplied).
3. From the **unzipped project root**, initialize Git and upload *the contents* as repository files:

```bash
git init
git add .
git commit -m "Release DC-PANR single-method receiver v1.0.0"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/DC-PANR.git
git push -u origin main
```

`YOUR_USERNAME` is your GitHub user name. Git will require appropriate login/authentication; do not put a password or token in the repository or in messages.

**Alternative with authenticated GitHub CLI** (after `git init`, `git add`, `git commit`):

```bash
gh repo create DC-PANR --public --source=. --remote=origin --push
```

4. Inspect the GitHub file tree and README rendering. Run the offered Actions smoke-test workflow and verify it passes.
5. In the GitHub repository, open `Releases` > `Draft a new release`, create tag `v1.0.0`, set a description, and publish. A release page is a stable reference; it is not a journal performance claim.

## Optional software DOI via Zenodo

1. Sign in at <https://zenodo.org/> using GitHub and authorize integration.
2. At Zenodo's `GitHub` settings, enable the public `DC-PANR` repository **before making a new release**.
3. Publish a new GitHub release (e.g. `v1.0.0`) and wait for Zenodo to archive it and assign the **software DOI**.
4. Add the resulting Zenodo DOI badge to `README.md` after it actually exists. The paper DOI and software DOI are different and should not be confused.

## Suggested GitHub description

`Official code for Direction-Conditioned Pilot-Aided Neural Receiver for Cochannel Signal Separation (IEEE Wireless Communications Letters, 2026). Single-model training and pilot-aided inference.`

## Topics

`signal-separation`, `wireless-communications`, `neural-receiver`, `pytorch`, `cochannel-interference`, `pilot-aided`, `direction-of-arrival`, `complex-valued-neural-network`

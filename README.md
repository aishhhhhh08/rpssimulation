# State / Militia / Intelligentsia — Convergence Simulator (web app)

A standalone web page version of the simulation. Same model, same math,
same visuals as the notebook -- just packaged so you can share **one link**
instead of a Colab notebook, and open that link directly in a browser
(works fine on a classroom projector or any student's laptop, no login,
no Colab account needed).

## Files in this folder
- `app.py` -- the Streamlit web app itself (thin wrapper around the two files below)
- `sim_engine.py` -- the simulation engine (identical to the notebook version)
- `sim_visualize.py` -- the dashboard/animation builder (identical to the notebook version)
- `requirements.txt` -- the three packages needed to run it

## Get a shareable link (recommended — takes about 5 minutes, free)

Streamlit Community Cloud will host this for you and give you a permanent
public URL like `https://your-app-name.streamlit.app` that opens directly
in any browser — this is the link you'd put on a slide or share with a class.

1. Create a free GitHub account if you don't have one (github.com).
2. Create a new GitHub repository and upload all 4 files in this folder
   (`app.py`, `sim_engine.py`, `sim_visualize.py`, `requirements.txt`) --
   they must all sit in the same top-level folder of the repo.
3. Go to **share.streamlit.io**, sign in with your GitHub account.
4. Click **New app**, pick your repository, set the main file path to
   `app.py`, and click **Deploy**.
5. Wait about a minute for it to build. You'll get a public link you can
   open on any device, share by URL, or embed in a slide.

Every time you push a change to that GitHub repo, the live link updates
automatically -- no redeploying by hand.

## Run it locally instead (no link, just on your own machine)

```
pip install -r requirements.txt
streamlit run app.py
```

This opens `http://localhost:8501` in your browser. Fine for testing on
your own laptop, but nobody else can open that link -- for a real
shareable URL, use the Streamlit Community Cloud steps above.

## What's in the app

The sidebar mirrors the notebook's control panel exactly:
- All 9 pre-condition cards, each with its own on/off checkbox and a
  typeable weight box (default 1.0, no normalization -- mix and match freely).
- Years to simulate, both convergence thresholds (individual win % /
  combined win %), playback speed, random seed, starting strengths for
  each actor, and the two global `Precondition wt` / `Legitimacy wt` sliders.

Press **Run simulation** and the main panel shows the same four-part
animated dashboard as the notebook (stacked-area power chart, live public
network, strength trace, real-time narration strip), followed by the
same written end-of-run explanation.

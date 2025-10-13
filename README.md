# Crying Face Filter (Emotion-Triggered, Realistic)

A lightweight, mobile-friendly web filter that turns any face into a sad, crying face using MediaPipe FaceMesh, with emotion-triggered tears. Works on iOS Safari and Android Chrome and can be deployed via GitHub Pages.

## Demo
- Deploy your repo as GitHub Pages and visit: `https://<your-username>.github.io/crying-face-filter/`

## Features
- Realistic, emotion-triggered tears (lowered brows + downturned mouth)
- Smooth, scale-invariant expression detection
- Selfie mirroring, camera flip button
- Web Share API capture (saves directly to Photos when supported), fallback to download

## Quick Start
1. Upload this folder to a new GitHub repo (e.g., `crying-face-filter`).
2. Enable Settings -> Pages -> Deploy from branch -> main / root.
3. Open the GitHub Pages URL on your phone.
4. Grant camera permission when prompted.
5. Make a sad face - tears appear when the trigger is met.
6. Tap Take Photo to save/share.

## Notes
- Uses MediaPipe from CDN; requires internet to load libraries.
- Some devices require a user gesture (tap) before the camera starts; we attempt auto-start.
- Emotion thresholds are tuned conservatively; adjust in `script.js` (`sad > 0.55`, EAR, etc.).
- If you want continuous tears, draw streaks unconditionally or reduce the threshold.

## Local Dev
Open `index.html` from a local server (CORS). Example using Python:
```bash
python3 -m http.server 8080
# then open http://localhost:8080
```

## QR Code
A placeholder `qr_code.png` is included. After you deploy, regenerate a real QR pointing to your GitHub Pages URL using any QR generator, or run:
```bash
# if you have Python 'qrcode' library installed
python - << 'PY'
import qrcode
img = qrcode.make('https://<your-username>.github.io/crying-face-filter/')
img.save('qr_code.png')
PY
```

## License
MIT

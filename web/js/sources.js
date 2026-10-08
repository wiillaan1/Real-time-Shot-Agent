// Video sources (browser side). Design doc 2.6: the intake layer hands over only "one image + a timestamp + capabilities".
//
// Every source implements the same four methods, and main.js does not care what is behind them:
//   open()        open it (ask for permission here if needed)
//   caps()        capabilities: kind, frame size, sensors; registered with /api/source
//   grab(upload)  take one frame: { blob: JPEG, ts: capture time (seconds), sensors: readings that go with the frame, or null }
//                 returns null when no picture is available yet; the frame loop tries again shortly
//   close()
//
// The server-side counterpart is shotagent/frame_source.py. To add a source, write another class of this shape.

export class WebcamSource {
  constructor(video) {
    this.video = video;
    this.canvas = document.createElement('canvas');
    this.stream = null;
  }

  async open() {
    if (!navigator.mediaDevices?.getUserMedia) {
      // Browsers only expose the camera over https or on localhost
      throw new Error('This address cannot use the camera: open the page at http://localhost:<port> or over https');
    }
    this.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    this.video.srcObject = this.stream;
    await this.video.play();
    if (!this.video.videoWidth) {
      await new Promise((resolve) => this.video.addEventListener('loadedmetadata', resolve, { once: true }));
    }
  }

  caps() {
    const track = this.stream.getVideoTracks()[0];
    return {
      kind: 'webcam',
      label: track.label || 'Camera',
      width: this.video.videoWidth,
      height: this.video.videoHeight,
      fps: track.getSettings().frameRate || 0,
      sensors: [],          // a laptop webcam has no gyro: sensor-based constraints degrade to text prompts
    };
  }

  aspect() {
    return this.video.videoWidth / this.video.videoHeight;
  }

  async grab(upload) {
    if (!this.video.videoWidth || !this.video.videoHeight) return null;   // the camera has no picture yet
    const width = Math.min(upload.width, this.video.videoWidth);
    const height = Math.round(width / this.aspect());
    this.canvas.width = width;
    this.canvas.height = height;
    this.canvas.getContext('2d').drawImage(this.video, 0, 0, width, height);
    const blob = await new Promise((resolve) => this.canvas.toBlob(resolve, 'image/jpeg', upload.jpeg_quality));
    if (!blob) return null;
    return { blob, ts: Date.now() / 1000, sensors: null };
  }

  close() {
    this.stream?.getTracks().forEach((track) => track.stop());
    this.video.srcObject = null;
    this.stream = null;
  }
}

// Simulated picture: runs the whole loop when there is no camera and no model.
// The server renders the picture (/api/sim/frame) and it is sent back like a real frame;
// rendering and recognising are one file on the server (perception/synthetic.py), so the two always agree.
// It declares a gyro and sends the "Camera pitch" slider's value along as the reading.
export class SimSource {
  constructor(image, readScene) {
    this.image = image;
    this.readScene = readScene;   // () => { table, bag, bagUnder, person, facing, pitch }
  }

  async open() {}

  caps() {
    return { kind: 'synthetic', label: 'Simulated', width: 640, height: 360, fps: 0, sensors: ['gyro'] };
  }

  aspect() {
    return 640 / 360;
  }

  async grab() {
    const scene = this.readScene();
    const query = new URLSearchParams({
      table: scene.table, bag_under: scene.bagUnder, facing: scene.facing, pitch: scene.pitch,
    });
    if (scene.bag !== null) query.set('bag', scene.bag);
    if (scene.person !== null) query.set('person', scene.person);

    const response = await fetch(`/api/sim/frame?${query}`);
    if (!response.ok) throw new Error(`Rendering the simulated picture failed (${response.status})`);
    const blob = await response.blob();

    const previous = this.image.src;
    this.image.src = URL.createObjectURL(blob);
    if (previous.startsWith('blob:')) URL.revokeObjectURL(previous);
    return { blob, ts: Date.now() / 1000, sensors: { pitch: scene.pitch } };
  }

  close() {}
}

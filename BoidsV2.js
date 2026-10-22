/**
 * 3D flocking simulation. Each boid combines separation, alignment, and cohesion; instanced meshes draw the flock efficiently.
 */
import * as THREE from 'three';
import { OrbitControls } from '../../Lib/Three/OrbitControls.js';

// Local flocking rules: https://www.red3d.com/cwr/boids/
const settings = { count: 220, separation: 1.6, alignment: 1.0, cohesion: 0.85, speed: 12 };
// Keep the population control and GPU instance allocation within the same limit.
const MAX_BOIDS = 2000;
const WORLD_RADIUS = 42;
const NEIGHBOR_RADIUS = 12;
const SEPARATION_RADIUS = 4;
const MAX_FORCE = 12;
const STEP = 1 / 60;
let paused = false;
const mount = document.createElement('main');
mount.id = 'boids-view';
document.body.appendChild(mount);
const panel = document.createElement('section');
panel.id = 'boids-controls';
panel.setAttribute('aria-label', 'Flock controls');
panel.innerHTML = `<h1>Boids <span>V2</span></h1><p>Three simple rules. One living flock.</p><div id="sliders"></div><div class="boids-actions"><button id="pause-boids" type="button">Pause</button><button id="reset-boids" type="button">Reset flock</button></div><small>Drag to orbit · Scroll to zoom</small><a href="https://www.red3d.com/cwr/boids/" target="_blank" rel="noopener">About the flocking model ↗</a>`;
mount.appendChild(panel);
for (const [key, label, min, max, step] of [
  ['count', 'Boids', 30, MAX_BOIDS, 10],
  ['separation', 'Separation', 0, 3, 0.1],
  ['alignment', 'Alignment', 0, 3, 0.1],
  ['cohesion', 'Cohesion', 0, 3, 0.1],
  ['speed', 'Speed', 3, 24, 1],
]) {
  const row = document.createElement('label');
  row.innerHTML = `<span>${label}</span><output>${settings[key]}</output><input aria-label="${label}" type="range" min="${min}" max="${max}" step="${step}" value="${settings[key]}">`;
  row.querySelector('input').addEventListener('input', (event) => {
    settings[key] = Number(event.target.value);
    row.querySelector('output').value = settings[key];
    if (key === 'count') reset();
  });
  panel.querySelector('#sliders').appendChild(row);
}

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x080f1d);
scene.fog = new THREE.Fog(0x080f1d, 130, 260);
const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 400);
camera.position.set(80, 48, 90);
let renderer;
try {
  renderer = new THREE.WebGLRenderer({ antialias: true });
} catch (error) {
  const message = document.createElement('p');
  message.className = 'boids-error';
  message.textContent = 'The simulation needs a browser with WebGL enabled.';
  mount.appendChild(message);
  throw error;
}
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.domElement.setAttribute('aria-label', 'Animated three-dimensional flock of boids');
mount.prepend(renderer.domElement);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.minDistance = 25;
controls.maxDistance = 200;
scene.add(new THREE.HemisphereLight(0xc5efff, 0x19334b, 2.4));
const light = new THREE.DirectionalLight(0xffffff, 2);
light.position.set(20, 50, 30);
scene.add(light);
const grid = new THREE.GridHelper(110, 22, 0x24465f, 0x13283d);
grid.position.y = -WORLD_RADIUS - 5;
scene.add(grid);

// Cone geometry points along +Y; every instance faces its own velocity.
const geometry = new THREE.ConeGeometry(0.65, 2.5, 4);
geometry.translate(0, 0.6, 0);
const material = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.45, metalness: 0.15 });
const flock = new THREE.InstancedMesh(geometry, material, MAX_BOIDS);
flock.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
flock.frustumCulled = false;
scene.add(flock);
const boids = [];
const dummy = new THREE.Object3D();
const up = new THREE.Vector3(0, 1, 0);
const direction = new THREE.Vector3();
const offset = new THREE.Vector3();
const separation = new THREE.Vector3();
const alignment = new THREE.Vector3();
const cohesion = new THREE.Vector3();
const steering = new THREE.Vector3();
const color = new THREE.Color();

/**
 * Rebuild flock state and instance colors when the population changes or the user resets.
 *
 * Inputs: None; uses enclosing state where needed.
 * @returns No value is returned; updates the current state, scene, or interface.
 */
function reset() {
  boids.length = 0;
  for (let i = 0; i < settings.count; i++) {
    boids.push({
      position: new THREE.Vector3().randomDirection().multiplyScalar(Math.cbrt(Math.random()) * 32),
      velocity: new THREE.Vector3().randomDirection().multiplyScalar(settings.speed * (0.6 + Math.random() * 0.4)),
      acceleration: new THREE.Vector3(),
    });
    color.setHSL(0.46 + Math.random() * 0.13, 0.8, 0.55 + Math.random() * 0.15);
    flock.setColorAt(i, color);
  }
  flock.count = settings.count;
  flock.instanceColor.needsUpdate = true;
  drawBoids();
}

// Turn a desired direction into bounded acceleration.
/**
 * Convert a desired direction into acceleration bounded by the steering-force limit.
 *
 * @param vector - Desired steering direction, updated in place to contain bounded acceleration.
 * @param velocity - Current boid velocity used to compute the required steering acceleration.
 * @returns The input steering vector updated in place with bounded acceleration.
 */
function steer(vector, velocity) {
  if (vector.lengthSq() > 0) vector.setLength(settings.speed).sub(velocity).clampLength(0, MAX_FORCE);
  return vector;
}

/**
 * Accumulate steering forces first, then integrate velocities and positions with a fixed time step.
 *
 * @param dt - Elapsed simulation time in seconds.
 * @returns No value is returned; updates the current state, scene, or interface.
 */
function simulate(dt) {
  // Compute all accelerations from the same snapshot before moving any boid.
  for (const boid of boids) {
    separation.set(0, 0, 0);
    alignment.set(0, 0, 0);
    cohesion.set(0, 0, 0);
    let neighbors = 0;
    for (const other of boids) {
      if (other === boid) continue;
      offset.subVectors(boid.position, other.position);
      const distanceSq = offset.lengthSq();
      if (distanceSq < NEIGHBOR_RADIUS ** 2) {
        alignment.add(other.velocity);
        cohesion.add(other.position);
        neighbors++;
        if (distanceSq < SEPARATION_RADIUS ** 2) {
          // Inverse distance weighting gives the closest neighbors priority.
          separation.addScaledVector(offset, 1 / Math.max(distanceSq, 0.001));
        }
      }
    }
    boid.acceleration.set(0, 0, 0);
    if (neighbors) {
      cohesion.divideScalar(neighbors).sub(boid.position);
      alignment.divideScalar(neighbors);
      boid.acceleration.addScaledVector(steer(separation, boid.velocity), settings.separation);
      boid.acceleration.addScaledVector(steer(alignment, boid.velocity), settings.alignment);
      boid.acceleration.addScaledVector(steer(cohesion, boid.velocity), settings.cohesion);
    }
    // A soft spherical boundary turns the flock back toward the scene.
    const radius = boid.position.length();
    if (radius > WORLD_RADIUS - 10) {
      steering.copy(boid.position).negate();
      boid.acceleration.addScaledVector(steer(steering, boid.velocity), (radius - WORLD_RADIUS + 10) / 5);
    }
    boid.acceleration.clampLength(0, MAX_FORCE * 3);
  }
  for (const boid of boids) {
    boid.velocity.addScaledVector(boid.acceleration, dt);
    boid.velocity.clampLength(settings.speed * 0.5, settings.speed);
    boid.position.addScaledVector(boid.velocity, dt);
  }
}

/**
 * Copy positions and velocity-facing rotations into the shared instanced mesh.
 *
 * Inputs: None; uses enclosing state where needed.
 * @returns No value is returned; updates the current state, scene, or interface.
 */
function drawBoids() {
  boids.forEach((boid, index) => {
    dummy.position.copy(boid.position);
    direction.copy(boid.velocity).normalize();
    dummy.quaternion.setFromUnitVectors(up, direction);
    dummy.updateMatrix();
    flock.setMatrixAt(index, dummy.matrix);
  });
  flock.instanceMatrix.needsUpdate = true;
}

panel.querySelector('#pause-boids').addEventListener('click', (event) => {
  paused = !paused;
  event.target.textContent = paused ? 'Resume' : 'Pause';
  event.target.setAttribute('aria-pressed', String(paused));
});
panel.querySelector('#reset-boids').addEventListener('click', reset);
/**
 * Synchronize renderer dimensions and camera projection with the available container size.
 *
 * Inputs: None; uses enclosing state where needed.
 * @returns No value is returned; updates the current state, scene, or interface.
 */
const resize = () => {
  const { width, height } = mount.getBoundingClientRect();
  camera.aspect = width / Math.max(height, 1);
  camera.updateProjectionMatrix();
  renderer.setSize(width, height);
};
// Recalculate view dimensions when the surrounding layout changes.
new ResizeObserver(resize).observe(mount);
resize();
reset();
let lastTime;
let accumulator = 0;
// Keep visual frames separate from the simulation's own update timing.
renderer.setAnimationLoop((time) => {
  const delta = lastTime === undefined ? 0 : Math.min((time - lastTime) / 1000, 0.05);
  lastTime = time;
  if (!paused && !document.hidden) {
    accumulator += delta;
    while (accumulator >= STEP) {
      simulate(STEP);
      accumulator -= STEP;
    }
    drawBoids();
  } else accumulator = 0;
  controls.update();
  renderer.render(scene, camera);
});

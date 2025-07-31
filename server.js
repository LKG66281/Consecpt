const express = require('express');
const fs = require('fs').promises;
const path = require('path');
const app = express();
const http = require('http').createServer(app);
const io = require('socket.io')(http);

// Parameters
const n = 50, d_v = 5, d_s = 3, d_t = 10, d = d_v + d_s + d_t, v = 100, a = 4;
const eta = 0.01, alpha = 0.9, gamma = 0.9, T = 100;
const theta_W = 0.01, theta_V = 0.01, theta_M = 0.01, theta_Q = 0.01, delta_e = 0.05;
const boxSize = 200;

// State initialization
let W = Array(n).fill().map(() => Array(d).fill().map(() => Math.random() * 0.02 - 0.01));
let V = Array(n).fill().map(() => Array(n).fill().map(() => Math.random() * 0.02 - 0.01));
let M = Array(n).fill().map(() => Array(v).fill(0));
let Q = Array(n).fill().map(() => Array(a).fill(0));
let A = Array(n).fill(0);
let E = [0.5, 0.5, 0.5];
let G = [];
let t = 0;
let P_triangle = [100, 100];
let P_ball = [150, 150];

// Vocabulary
const V_vocab = Array(v).fill().map((_, i) => {
  if (i === 0) return "Hello!";
  if (i === 1) return "Blue due to scattering";
  if (i === 2) return "I feel you";
  if (i === 3) return "Thinking...";
  return `Token${i}`;
});

// Persistence
const stateFile = path.join(__dirname, 'state.json');
const saveState = async () => {
  try {
    const state = { W, V, M, Q, A, E, G, t, P_triangle, P_ball };
    await fs.writeFile(stateFile, JSON.stringify(state, null, 2));
  } catch (err) {
    console.error('Error saving state:', err);
  }
};

const loadState = async () => {
  try {
    const data = await fs.readFile(stateFile, 'utf8');
    const state = JSON.parse(data);
    W = state.W || W;
    V = state.V || V;
    M = state.M || M;
    Q = state.Q || Q;
    A = state.A || A;
    E = state.E || E;
    G = state.G || G;
    t = state.t || t;
    P_triangle = state.P_triangle || P_triangle;
    P_ball = state.P_ball || P_ball;
  } catch (err) {
    console.log('No state file found, using default state');
  }
};

loadState();

// Math functions
const sigmoid = x => 1 / (1 + Math.exp(-x));
const matMul = (A, B) => {
  const result = Array(A.length).fill().map(() => Array(B[0].length).fill(0));
  for (let i = 0; i < A.length; i++)
    for (let j = 0; j < B[0].length; j++)
      for (let k = 0; k < B.length; k++)
        result[i][j] += A[i][k] * B[k][j];
  return result;
};

// System updates
const updateActivations = S => {
  const sensory = matMul(W, [S])[0];
  const assoc = matMul(V, [A])[0];
  A = A.map((a, i) => sigmoid(alpha * a + (1 - alpha) * (sensory[i] + assoc[i])));
};

const updateSensoryWeights = S => {
  for (let i = 0; i < n; i++)
    for (let m = 0; m < d; m++)
      W[i][m] += eta * A[i] * (S[m] - W[i][m]);
};

const updateAssociationWeights = A_prev => {
  for (let i = 0; i < n; i++)
    for (let k = 0; k < n; k++)
      V[i][k] += eta * A_prev[i] * A[k];
};

const updateVocabularyWeights = T => {
  for (let i = 0; i < n; i++)
    for (let j = 0; j < v; j++)
      M[i][j] += eta * A[i] * (T[j] - M[i][j]);
};

const updateActionWeights = action, reward => {
  const Q_target = reward + gamma * Math.max(...matMul([A], Q)[0]);
  for (let i = 0; i < n; i++)
    Q[i][action] += eta * A[i] * (Q_target - Q[i][action]);
};

const pruneWeights = () => {
  for (let i = 0; i < n; i++) {
    for (let m = 0; m < d; m++) if (Math.abs(W[i][m]) < theta_W) W[i][m] = 0;
    for (let k = 0; k < n; k++) if (Math.abs(V[i][k]) < theta_V) V[i][k] = 0;
    for (let j = 0; j < v; j++) if (Math.abs(M[i][j]) < theta_M) M[i][j] = 0;
    for (let j = 0; j < a; j++) if (Math.abs(Q[i][j]) < theta_Q) Q[i][j] = 0;
  }
};

const computeConfidence = () => {
  const sumA = A.reduce((sum, a) => sum + a, 0) || 1;
  const P = A.map(a => a / sumA);
  const H = -P.reduce((sum, p) => sum + (p > 0 ? p * Math.log(p) : 0), 0);
  return 1 - H / Math.log(n);
};

const updateEmotions = (S_text, touch) => {
  const sumText = S_text.reduce((sum, s) => sum + s, 0);
  const f_interaction = touch ? 1 : sumText > 0.5 * d_t ? 1 : sumText < 0.2 * d_t ? -1 : 0;
  E = E.map(e => Math.max(0, Math.min(1, e + delta_e * f_interaction)));
};

const textToBinary = input => {
  return Array(d_t).fill().map((_, i) => (input.length + i) % (i + 2) > i / 2 ? 1 : 0);
};

const textToVocabIndex = input => {
  if (input.toLowerCase().includes('hey bro')) return 0;
  if (input.toLowerCase().includes('why is the sky blue')) return 1;
  if (input.toLowerCase().includes('sad')) return 2;
  return -1;
};

// Visual state
const moves = [[0, -5], [0, 5], [-5, 0], [5, 0]]; // Up, down, left, right
const updateTriangle = () => {
  const actionScores = matMul([A], Q)[0];
  const action = actionScores.indexOf(Math.max(...actionScores));
  const [dx, dy] = moves[action];
  P_triangle[0] = Math.max(0, Math.min(boxSize, P_triangle[0] + dx));
  P_triangle[1] = Math.max(0, Math.min(boxSize, P_triangle[1] + dy));
  return action;
};

const updateBall = touch => {
  if (touch) {
    P_ball = [Math.random() * boxSize, Math.random() * boxSize];
  } else {
    P_ball[0] = Math.max(0, Math.min(boxSize, P_ball[0] + (Math.random() - 0.5) * 4));
    P_ball[1] = Math.max(0, Math.min(boxSize, P_ball[1] + (Math.random() - 0.5) * 4));
  }
};

const checkTouch = () => {
  const dx = P_triangle[0] - P_ball[0];
  const dy = P_triangle[1] - P_ball[1];
  return Math.sqrt(dx * dx + dy * dy) < 20; // Triangle and ball radius
};

const generateResponse = async (S_text, input) => {
  const C = computeConfidence();
  const maxA = Math.max(...A);
  const activeConcept = A.indexOf(maxA);
  let responseBits, T = Array(v).fill(0);
  const vocabIndex = textToVocabIndex(input);
  if (vocabIndex >= 0) T[vocabIndex] = 1;

  if (S_text.reduce((sum, s) => sum + s, 0) > 0.5 * d_t) {
    responseBits = [1, 0, 1, 0];
    G.push([`concept${activeConcept}`, 'is', 'greeting']);
    T[0] = 1;
  } else if (G.length > 0 && maxA > 0.7) {
    responseBits = [1, 1, 0, 0];
    G.push([`concept${activeConcept}`, 'has_property', `prop${t}`]);
    T[1] = 1;
  } else if (E[1] > 0.6) {
    responseBits = [0, 0, 1, 1];
    T[2] = 1;
  } else if (E[0] > 0.6 && G.length > 0) {
    responseBits = [1, 1, 1, 1];
    T[1] = 1;
  } else {
    responseBits = [0, 0, 0, 0];
    T[3] = 1;
  }

  const action = updateTriangle();
  const touch = checkTouch();
  updateBall(touch);
  updateEmotions(S_text, touch);
  updateVocabularyWeights(T);
  updateActionWeights(action, touch ? 1 : 0);
  const vocabActivations = matMul([A], M)[0];
  const vocabIndexOut = vocabActivations.indexOf(Math.max(...vocabActivations));
  const englishResponse = C < 0.5 ? `Thinking... ${V_vocab[vocabIndexOut]}` : V_vocab[vocabIndexOut];
  await saveState();
  return { response: responseBits.join(''), english: englishResponse, emotions: E, confidence: C, P_triangle, P_ball, touch };
};

// Animation loop (runs even without users)
setInterval(async () => {
  const action = updateTriangle();
  const touch = checkTouch();
  updateBall(touch);
  updateEmotions(Array(d_t).fill(0), touch);
  updateActionWeights(action, touch ? 1 : 0);
  await saveState();
  io.emit('visualUpdate', { P_triangle, P_ball, emotions: E });
}, 100);

// Serve static files
app.use(express.static(path.join(__dirname, 'public')));

// Socket.IO
io.on('connection', socket => {
  console.log('User connected');
  socket.emit('visualUpdate', { P_triangle, P_ball, emotions: E });
  socket.on('userInput', async ({ input, vision, sound }) => {
    const A_prev = [...A];
    const S_text = textToBinary(input);
    const S = [...vision, ...sound, ...S_text];
    updateActivations(S);
    updateSensoryWeights(S);
    updateAssociationWeights(A_prev);
    if (t++ % T === 0) pruneWeights();
    const responseData = await generateResponse(S_text, input);
    io.emit('response', responseData);
    io.emit('visualUpdate', { P_triangle: responseData.P_triangle, P_ball: responseData.P_ball, emotions: responseData.emotions });
  });
});

// Start server
const PORT = process.env.PORT || 3000;
http.listen(PORT, () => console.log(`Server running on port ${PORT}`));

const socket = io();
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

function drawFace(emotions) {
  const [happiness, sadness, frustration] = emotions;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Face background
  ctx.fillStyle = `rgb(${frustration * 255}, ${happiness * 255}, ${sadness * 255})`;
  ctx.beginPath();
  ctx.arc(150, 100, 50, 0, Math.PI * 2);
  ctx.fill();

  // Eyes
  const eyeSize = 10 + happiness * 10;
  ctx.fillStyle = 'black';
  ctx.beginPath();
  ctx.arc(130, 80, eyeSize, 0, Math.PI * 2);
  ctx.arc(170, 80, eyeSize, 0, Math.PI * 2);
  ctx.fill();

  // Mouth
  ctx.beginPath();
  ctx.moveTo(130, 100 + sadness * 20);
  ctx.quadraticCurveTo(150, 120 - happiness * 20, 170, 100 + sadness * 20);
  ctx.stroke();
}

function drawBox(P_triangle, P_ball, response) {
  // Box
  ctx.strokeStyle = 'black';
  ctx.strokeRect(50, 150, 200, 200);

  // Triangle (hand)
  ctx.fillStyle = 'green';
  ctx.beginPath();
  const angle = response === '1010' ? 0 : response === '1100' ? Math.PI / 2 : response === '0011' ? Math.PI : response === '1111' ? -Math.PI / 2 : 0;
  ctx.moveTo(P_triangle[0] + 50 + 10 * Math.cos(angle), P_triangle[1] + 150 + 10 * Math.sin(angle));
  ctx.lineTo(P_triangle[0] + 50 + 10 * Math.cos(angle + 2 * Math.PI / 3), P_triangle[1] + 150 + 10 * Math.sin(angle + 2 * Math.PI / 3));
  ctx.lineTo(P_triangle[0] + 50 + 10 * Math.cos(angle + 4 * Math.PI / 3), P_triangle[1] + 150 + 10 * Math.sin(angle + 4 * Math.PI / 3));
  ctx.closePath();
  ctx.fill();

  // Ball
  ctx.fillStyle = 'red';
  ctx.beginPath();
  ctx.arc(P_ball[0] + 50, P_ball[1] + 150, 10, 0, Math.PI * 2);
  ctx.fill();
}

socket.on('response', ({ response, english, emotions, confidence, P_triangle, P_ball }) => {
  const responseText = response === '1010' ? 'Positive' :
                       response === '1100' ? 'Reasoning' :
                       response === '0011' ? 'Sad' :
                       response === '1111' ? 'Happy Reasoning' : 'Uncertain';
  document.getElementById('response').textContent = `Binary Response: ${response} (${responseText})`;
  document.getElementById('english').textContent = `English Response: ${english}`;
  document.getElementById('emotions').textContent = `Emotions: Happiness=${emotions[0].toFixed(2)}, Sadness=${emotions[1].toFixed(2)}, Frustration=${emotions[2].toFixed(2)}`;
  document.getElementById('confidence').textContent = `Confidence: ${confidence.toFixed(2)}`;
  drawFace(emotions);
  drawBox(P_triangle, P_ball, response);
});

socket.on('visualUpdate', ({ P_triangle, P_ball, emotions }) => {
  drawFace(emotions);
  drawBox(P_triangle, P_ball, document.getElementById('response').textContent.split(' ')[2] || '0000');
});

function getVisionFeatures(P_triangle, P_ball) {
  const dx = P_ball[0] - P_triangle[0];
  const dy = P_ball[1] - P_triangle[1];
  return [
    dx > 0 ? 1 : 0,
    dx < 0 ? 1 : 0,
    dy > 0 ? 1 : 0,
    dy < 0 ? 1 : 0,
    Math.sqrt(dx * dx + dy * dy) < 20 ? 1 : 0
  ];
}

function getSoundFeatures() {
  return Array(3).fill().map(() => Math.random() > 0.5 ? 1 : 0);
}

function sendInput() {
  const input = document.getElementById('userInput').value.trim();
  if (!input) {
    alert('Please enter an input');
    return;
  }
  const vision = getVisionFeatures(P_triangle, P_ball);
  const sound = getSoundFeatures();
  socket.emit('userInput', { input, vision, sound });
}

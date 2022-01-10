const PI2 = 2 * Math.PI;

"use strict";


/* 
 * Computes the discrete Fourier transform (DFT) of the given complex vector, storing the result back into the vector.
 * The vector can have any length. This is a wrapper function.
 */
function transform(real, imag) {
	var n = real.length;
	if (n != imag.length)
		throw "Mismatched lengths";
	if (n == 0)
		return;
	else if ((n & (n - 1)) == 0)  // Is power of 2
		transformRadix2(real, imag);
	else  // More complicated algorithm for arbitrary sizes
		transformBluestein(real, imag);
}


/* 
 * Computes the inverse discrete Fourier transform (IDFT) of the given complex vector, storing the result back into the vector.
 * The vector can have any length. This is a wrapper function. This transform does not perform scaling, so the inverse is not a true inverse.
 */
function inverseTransform(real, imag) {
	transform(imag, real);
}


/* 
 * Computes the discrete Fourier transform (DFT) of the given complex vector, storing the result back into the vector.
 * The vector's length must be a power of 2. Uses the Cooley-Tukey decimation-in-time radix-2 algorithm.
 */
function transformRadix2(real, imag) {
	// Length variables
	var n = real.length;
	var hn = n / 2;
	var levels = -1;
	for (var i = 0; i < 32; ++i) {
		if (1 << i == n)
			levels = i;  // Equal to log2(n)
	}
	
	// Trigonometric tables
	var cosTable = new Array(hn);
	var sinTable = new Array(hn);
	for (var i = 0; i < hn; ++i) {
		var f = PI2 * i / n;
		cosTable[i] = Math.cos(f);
		sinTable[i] = Math.sin(f);
	}
	
	// Bit-reversed addressing permutation
	for (var i = 0; i < n; ++i) {
		var j = reverseBits(i, levels);
		if (j > i) {
			var temp = real[i];
			real[i] = real[j];
			real[j] = temp;
			temp = imag[i];
			imag[i] = imag[j];
			imag[j] = temp;
		}
	}
	
	// Cooley-Tukey decimation-in-time radix-2 FFT
	for (var size = 2; size <= n; size *= 2) {
		var halfsize = size / 2;
		var tablestep = n / size;
		for (var i = 0; i < n; i += size) {
			for (var j = i, k = 0; j < i + halfsize; j++, k += tablestep) {
				var l = j + halfsize;
				var tpre =  real[l] * cosTable[k] + imag[l] * sinTable[k];
				var tpim = -real[l] * sinTable[k] + imag[l] * cosTable[k];
				real[l] = real[j] - tpre;
				imag[l] = imag[j] - tpim;
				real[j] += tpre;
				imag[j] += tpim;
			}
		}
	}
	
	// Returns the integer whose value is the reverse of the lowest 'bits' bits of the integer 'x'.
	function reverseBits(x, bits) {
		var y = 0;
		for (var i = 0; i < bits; ++i) {
			y = (y << 1) | (x & 1);
			x >>>= 1;
		}
		return y;
	}
}


/* 
 * Computes the discrete Fourier transform (DFT) of the given complex vector, storing the result back into the vector.
 * The vector can have any length. This requires the convolution function, which in turn requires the radix-2 FFT function.
 * Uses Bluestein's chirp z-transform algorithm.
 */
function transformBluestein(real, imag) {
	// Find a power-of-2 convolution length m such that m >= n * 2 + 1
	var n = real.length;
	if (n != imag.length)
		throw "Mismatched lengths";
	var m = 1;
	while (m < n * 2 + 1)
		m *= 2;
	
	// Trignometric tables
	var cosTable = new Array(n);
	var sinTable = new Array(n);
	for (var i = 0; i < n; ++i) {
		var j = i * i % (n * 2);  // This is more accurate than j = i * i
		cosTable[i] = Math.cos(Math.PI * j / n);
		sinTable[i] = Math.sin(Math.PI * j / n);
	}
	
	// Temporary vectors and preprocessing
	var areal = new Array(m).fill(0);
	var aimag = new Array(m).fill(0);
	for (var i = 0; i < n; ++i) {
		areal[i] =  real[i] * cosTable[i] + imag[i] * sinTable[i];
		aimag[i] = -real[i] * sinTable[i] + imag[i] * cosTable[i];
	}
	var breal = new Array(m).fill(0);
	var bimag = new Array(m).fill(0);
	breal[0] = cosTable[0];
	bimag[0] = sinTable[0];
	for (var i = 1; i < n; ++i) {
		breal[i] = breal[m - i] = cosTable[i];
		bimag[i] = bimag[m - i] = sinTable[i];
	}
	
	// Convolution
	var creal = new Array(m);
	var cimag = new Array(m);
	convolveComplex(areal, aimag, breal, bimag, creal, cimag);
	
	// Postprocessing
	for (var i = 0; i < n; ++i) {
		real[i] =  creal[i] * cosTable[i] + cimag[i] * sinTable[i];
		imag[i] = -creal[i] * sinTable[i] + cimag[i] * cosTable[i];
	}
}


/* 
 * Computes the circular convolution of the given real vectors. Each vector's length must be the same.
 */
function convolveReal(x, y, out) {
	var n = x.length;
	if (n != y.length || n != out.length)
		throw "Mismatched lengths";
	convolveComplex(x, new Array(n).fill(0), y, new Array(n).fill(0), out, new Array(n).fill(0));
}


/* 
 * Computes the circular convolution of the given complex vectors. Each vector's length must be the same.
 */
function convolveComplex(xreal, ximag, yreal, yimag, outreal, outimag) {
	var n = xreal.length;
	if (n != ximag.length || n != yreal.length || n != yimag.length
			|| n != outreal.length || n != outimag.length)
		throw "Mismatched lengths";
	
	xreal = xreal.slice();
	ximag = ximag.slice();
	yreal = yreal.slice();
	yimag = yimag.slice();
	transform(xreal, ximag);
	transform(yreal, yimag);
	
	for (var i = 0; i < n; ++i) {
		var temp = xreal[i] * yreal[i] - ximag[i] * yimag[i];
		ximag[i] = ximag[i] * yreal[i] + xreal[i] * yimag[i];
		xreal[i] = temp;
	}
	inverseTransform(xreal, ximag);
	
	for (var i = 0; i < n; ++i) {  // Scaling (because this FFT implementation omits it)
		outreal[i] = xreal[i] / n;
		outimag[i] = ximag[i] / n;
	}
}

var properties = {
    schemeColor: '#999'
};

// A global object that can listen to property changes
window.wallpaperPropertyListener = {
    applyUserProperties: function(properties) {
        // Read scheme color
        if (properties.schemecolor) {
            var schemeColor = properties.schemecolor.value.split(' ');
            schemeColor = schemeColor.map(function(c) {
                return Math.ceil(c * 255);
            });
            properties.schemeColor = schemeColor;
        }
    }
};


const audioNFrequencies = 128;
const audioUpdatesPerSecond = 30;
const localLengthSec = 1 / 2;

const audioHNFrequencies = audioNFrequencies / 2;

var parent;
var audioCanvas;
var audioCanvasCtx;
var barWidth;
var barsX = [];

var audioSamples = [];
// var spectrogram = [];

var lastAudioArray = new Array(audioHNFrequencies).fill(0);

var fpsElement;
var localLengthH = Math.floor(localLengthSec * audioUpdatesPerSecond / 2);
var localLength = 2 * localLengthH + 1;
var localWeights = [];

{
	var localLengthHInc = localLengthH + 1;
	var localLengthHIncSq = localLengthHInc * localLengthHInc;
	var localLengthInc = localLength + 1;
	for (var i = 0; i < localLength; ++i) {
		// rectangular sliding average
		// localWeights[i] = 1 / localLength;

		// triangular sliding average
		// localWeights[i] = 2 * (1 - Math.abs(2 * (i + 1) / localLengthInc - 1)) / localLengthInc;
		
		// smooth sliding average
		// var x = i - localLengthH;
		// var sq2 = localLengthHIncSq - x * x;
		// localWeights[i] = 15 * sq2 * sq2 / (localLengthHInc * (16 * localLengthHIncSq * localLengthHIncSq - 1));
		
		// hanning
		var s = Math.sin(Math.PI * (i + 1) / localLengthInc);
		localWeights[i] = 2 * s * s / localLengthInc;
	}
}

var frameLength = 2 ** 8 + localLength; // minimum 2 ** 11 to capture all whole BPM

var done = false;
var started = false;
var startTime = 0;
var novelty = [];
var noveltyI = new Array(frameLength - localLength).fill(0);
var localAverages = [];
var audio = new Audio('click.wav');
var bpm = 0;
var freq = 0;
var averageps = 0;
var waitLength = 0;
var phase = 0;
var compression = 1;
var betterFreq = 0;

var audioRightFrequencies = [];
for (var i = 0; i < audioHNFrequencies; ++i) {
	audioRightFrequencies[i] = i + audioHNFrequencies;
}

function sleep(ms) {
	return new Promise(resolve => setTimeout(resolve, ms));
}

function onWindowResized() {
    audioCanvas.width = parent.offsetWidth;
    audioCanvas.height = parent.offsetHeight;
	barWidth = audioCanvas.width / audioNFrequencies;
	for (var i = 0; i < audioNFrequencies; ++i) {
		barsX[i] = barWidth * i;
	}
}

function wallpaperAudioListener(audioArray) {
	// updated 30 times per secon
	audioSamples = audioArray;
	if (!done) {
		if (!started) {
			startTime = Date.now();
			started = true;
		}
		for (var i = 0; i < audioHNFrequencies; ++i) {
			audioArray[i] = Math.log(1 + compression * (audioArray[i] + audioArray[audioRightFrequencies[i]]) / 2) / Math.log(1 + compression);
			// audioArray[i] = (audioArray[i] + audioArray[i + audioHNFrequencies]) / 2;
		}
		audioArray = audioArray.slice(0, audioHNFrequencies);
		if (novelty.length < frameLength) {
			novelty.push(audioArray.reduce((a, b, i) => a + Math.max(b - lastAudioArray[i], 0), 0));
			if (novelty.length > localLength) {
				localAverages.push(localWeights.reduce((a, b, i) => a + b * novelty[novelty.length - localLength + i]));
				if (localAverages.length > localLengthH) {
					novelty[novelty.length - localLength] = Math.max(novelty[novelty.length - localLength] - localAverages[localAverages.length - 1 - localLengthH], 0);
					// novelty[novelty.length - localLength] = 0;
					// localAverages[localAverages.length - 1 - localLengthH] = 0;
				}
			}
			// spectrogram.push(audioArray);
		} else {
			averageps = frameLength * 1000 / (Date.now() - startTime);
			// startTime += 1000 * (localLengthH + 1) / averageps;
			novelty = novelty.slice(localLengthH + 1, novelty.length - localLengthH);
			for (var i = novelty.length - localLengthH; i < novelty.length; ++i) {
				novelty[i] = Math.max(novelty[i] - localAverages[i - 1], 0);
				// novelty[i] = 0;
				// localAverages[i - 1] = 0;
			}
			var average = novelty.reduce((a, b) => a + b) / novelty.length;	
			var weights = [];
			var r = 8;
			for (var i = 0; i < novelty.length; ++i) {
				var sq = 2 * i + 1 - novelty.length;
				weights[i] = Math.exp(- (r * r * sq * sq / (8 * (novelty.length - 1) * (novelty.length - 1))));
			}
			novelty = novelty.map((x, i) => (x - average) * weights[i]);
			done = true;
			transformRadix2(novelty, noveltyI);
			var oldNovelty = novelty.slice();
			novelty = novelty.map((x, i) => Math.sqrt(x * x + noveltyI[i] * noveltyI[i]));
			{
				var max = 0;
				var tempoMin = 30;
				var tempoMax = 240;
				for (var i = Math.ceil((tempoMin / 60) / (averageps / novelty.length)); i < Math.floor((tempoMax / 60) / (averageps / novelty.length)); ++i) {
					if (novelty[i] > max) {
						freq = i;
						max = novelty[i];
					}
				}
			}
			betterFreq = freq + Math.log(novelty[freq + 1] / novelty[freq - 1]) / (2 * Math.log(novelty[freq] * novelty[freq] / (novelty[freq - 1] * novelty[freq + 1])));
			phase = Math.atan2(noveltyI[freq], oldNovelty[freq]) / (Math.PI);
			// startTime -= 1000 * phase / (betterFreq * averageps / novelty.length);
			bpm = 60 * betterFreq * averageps / novelty.length;
			waitLength = 1000 / (Math.round(bpm) / 60);
		}
		lastAudioArray = audioArray;
	} else {
		novelty = [];
		noveltyI.fill(0);
		localAverages = [];
		done = false;
		started = false;
	}
	// if ((Date.now() - startTime) % waitLength <= 37) {
		// audio.play();
	// }
    fpsElement.textContent = 'phase: ' + phase + ' sps: ' + averageps + ' bpm: ' + bpm + ' n: ' + novelty.length + ', a: ' + localAverages.length;
};

function run() {
    window.requestAnimationFrame(run);
	audioCanvasCtx.fillStyle = 'rgb(255,255,255)';
    audioCanvasCtx.fillRect(0, 0, audioCanvas.width, audioCanvas.height);
	
	var width = audioCanvas.width / (frameLength - localLength);
	var amplitude = audioCanvas.height / novelty.reduce((a, b) => Math.max(a, b));
	audioCanvasCtx.beginPath();
	audioCanvasCtx.moveTo(0, audioCanvas.height - novelty[0] * amplitude);
	for (var i = 1; i < novelty.length; ++i) {
		audioCanvasCtx.lineTo(width * i, audioCanvas.height - novelty[i] * amplitude);
	}
	audioCanvasCtx.stroke();

	// audioCanvasCtx.beginPath();
	// if (done) {
		// audioCanvasCtx.moveTo(0, audioCanvas.height - localAverages[0] * amplitude);
		// for (var i = 1; i < localAverages.length; ++i) {
			// audioCanvasCtx.lineTo(width * i, audioCanvas.height - localAverages[i] * amplitude);
		// }
	// } else {
		// audioCanvasCtx.moveTo(width * (localLengthH + 1), audioCanvas.height - localAverages[0] * amplitude);
		// for (var i = 1; i < localAverages.length; ++i) {
			// audioCanvasCtx.lineTo(width * (i + localLengthH + 1), audioCanvas.height - localAverages[i] * amplitude);
		// }
	// }
	// audioCanvasCtx.stroke();
	
	// if (!done) {
		// var width = audioCanvas.width / spectrogram.length;
		// var height = audioCanvas.height / spectrogram[0].length;
		// var a = 255 / spectrogram.map(x => x.reduce((a, b) => Math.max(a, b))).reduce((a, b) => Math.max(a, b));
		// for (var i = 1; i < spectrogram.length; ++i) {
			// for (var j = 0; j < spectrogram[0].length; ++j) {
				// var amplitude = Math.round(a * spectrogram[i][j]);
				// audioCanvasCtx.fillStyle = 'rgb(' + amplitude + ',' + amplitude + ',' + amplitude + ')';
				// audioCanvasCtx.fillRect(width * i, audioCanvas.height - height * j, width, height);
			// }
			// audioCanvasCtx.stroke();
		// }
	// }
	
	
	
	
	// audioCanvasCtx.fillStyle = 'rgb(255,0,0)';
    // for (var i = 0; i < audioHNFrequencies; ++i) {
        // var height = audioCanvas.height * audioSamples[i];
        // audioCanvasCtx.fillRect(barsX[i], audioCanvas.height - height, barWidth, height);
    // }
	
    // audioCanvasCtx.fillStyle = 'rgb(0,0,255)';
    // for (var i = audioHNFrequencies; i < audioSamples.length; ++i) {
        // var height = audioCanvas.height * audioSamples[191 - i];
        // audioCanvasCtx.fillRect(barsX[i], audioCanvas.height - height, barWidth, height);
    // }
}

window.onload = function() {
    parent = document.getElementById('AudioCanvasParent');
    audioCanvas = document.getElementById('AudioCanvas');
    audioCanvasCtx = audioCanvas.getContext('2d');
	
	fpsElement = document.getElementById('AudioDisplay');
	
	onWindowResized();
	window.wallpaperRegisterAudioListener(wallpaperAudioListener);
	window.requestAnimationFrame(run);
};

window.onresize = onWindowResized;

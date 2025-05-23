/**
 * MoveNet Pose Detection implementation using TensorFlow.js
 * Handles camera input, pose detection, and visualization
 */
class MoveNetPoseDetector {
    constructor() {
        // DOM elements
        this.video = document.getElementById('video');
        this.canvas = document.getElementById('canvas');
        this.ctx = this.canvas.getContext('2d');
        
        // Detection state
        this.detector = null;
        this.isRunning = false;
        this.stream = null;
        this.animationId = null;
        
        // Visualization options
        this.showSkeleton = true;
        this.showKeypoints = true;
        
        // Performance tracking
        this.frameCount = 0;
        this.lastTime = Date.now();
        this.fps = 0;
        
        // Keypoint connections for skeleton
        this.connections = [
            [0, 1], [0, 2], [1, 3], [2, 4], // Head
            [5, 6], [5, 7], [7, 9], [6, 8], [8, 10], // Arms
            [5, 11], [6, 12], [11, 12], // Torso
            [11, 13], [13, 15], [12, 14], [14, 16] // Legs
        ];
        
        // Keypoint names for labeling
        this.keypointNames = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ];
        
        this.initializeEventListeners();
        this.loadModel();
    }
    
    /**
     * Set up UI control event handlers
     */
    initializeEventListeners() {
        document.getElementById('startBtn').addEventListener('click', () => this.startCamera());
        document.getElementById('stopBtn').addEventListener('click', () => this.stopCamera());
        document.getElementById('toggleSkeleton').addEventListener('click', () => this.toggleSkeleton());
        document.getElementById('toggleKeypoints').addEventListener('click', () => this.toggleKeypoints());
        document.getElementById('modelSelect').addEventListener('change', () => this.changeModel());
    }
    
    /**
     * Load selected TensorFlow.js pose detection model
     */
    async loadModel() {
        this.updateStatus('Loading TensorFlow.js model...');
        
        try {
            const modelType = document.getElementById('modelSelect').value;
            let modelName;
            
            switch(modelType) {
                case 'lightning':
                case 'thunder':
                case 'multipose':
                    modelName = window.poseDetection.SupportedModels.MoveNet;
                    break;
                default:
                    modelName = window.poseDetection.SupportedModels.MoveNet;
            }
            
            // Configure detector based on selected model type
            const detectorConfig = {
                modelType: modelType === 'lightning' ? 'lite' : 
                          modelType === 'thunder' ? 'full' : 'multipose',
                enableSmoothing: true,
                multiPoseMaxDimension: 256,
                enableTracking: true,
                trackerType: 'boundingBox'
            };
            
            this.detector = await window.poseDetection.createDetector(modelName, detectorConfig);
            this.updateStatus(`Model loaded: MoveNet ${modelType}`);
        } catch (error) {
            console.error('Error loading model:', error);
            this.updateStatus('Error loading model. Using fallback...');
            
            // Fallback: try basic MoveNet
            try {
                this.detector = await window.poseDetection.createDetector(
                    window.poseDetection.SupportedModels.MoveNet
                );
                this.updateStatus('Fallback model loaded successfully');
            } catch (fallbackError) {
                this.updateStatus('Failed to load any model');
            }
        }
    }
    
    /**
     * Handle model change from dropdown
     */
    async changeModel() {
        if (this.isRunning) {
            this.stopCamera();
        }
        await this.loadModel();
    }
    
    /**
     * Initialize camera and start pose detection
     */
    async startCamera() {
        if (!this.detector) {
            this.updateStatus('Model not loaded yet');
            return;
        }
        
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480, facingMode: 'user' }
            });
            
            this.video.srcObject = this.stream;
            this.video.onloadedmetadata = () => {
                this.video.play();
                this.startDetection();
            };
            
            document.getElementById('startBtn').disabled = true;
            document.getElementById('stopBtn').disabled = false;
            
        } catch (error) {
            console.error('Error accessing camera:', error);
            this.updateStatus('Error: Could not access camera');
        }
    }
    
    /**
     * Stop camera and detection process
     */
    stopCamera() {
        this.isRunning = false;
        
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
        
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        document.getElementById('startBtn').disabled = false;
        document.getElementById('stopBtn').disabled = true;
        
        this.updateStatus('Camera stopped');
        this.updateStats(0, 0, 0);
    }
    
    /**
     * Start continuous pose detection loop
     */
    async startDetection() {
        this.isRunning = true;
        this.lastTime = Date.now();
        this.frameCount = 0;
        
        const detectFrame = async () => {
            if (!this.isRunning) return;
            
            try {
                const poses = await this.detector.estimatePoses(this.video);
                this.drawResults(poses);
                this.updatePerformanceStats();
                this.updatePoseInfo(poses);
                
            } catch (error) {
                console.error('Detection error:', error);
            }
            
            this.animationId = requestAnimationFrame(detectFrame);
        };
        
        detectFrame();
        this.updateStatus('Pose detection active');
    }
    
    /**
     * Render detected poses on canvas
     * @param {Array} poses - Detected pose data
     */
    drawResults(poses) {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        if (poses.length === 0) return;
        
        poses.forEach(pose => {
            const keypoints = pose.keypoints;
            
            // Draw skeleton connections
            if (this.showSkeleton) {
                this.ctx.strokeStyle = '#00FF00';
                this.ctx.lineWidth = 2;
                
                this.connections.forEach(([i, j]) => {
                    const kp1 = keypoints[i];
                    const kp2 = keypoints[j];
                    
                    // Only draw connections between high-confidence keypoints
                    if (kp1.score > 0.3 && kp2.score > 0.3) {
                        this.ctx.beginPath();
                        this.ctx.moveTo(kp1.x, kp1.y);
                        this.ctx.lineTo(kp2.x, kp2.y);
                        this.ctx.stroke();
                    }
                });
            }
            
            // Draw keypoints with confidence-based coloring
            if (this.showKeypoints) {
                keypoints.forEach((keypoint, index) => {
                    if (keypoint.score > 0.3) {
                        this.ctx.fillStyle = keypoint.score > 0.7 ? '#FF0000' : '#FFFF00';
                        this.ctx.beginPath();
                        this.ctx.arc(keypoint.x, keypoint.y, 4, 0, 2 * Math.PI);
                        this.ctx.fill();
                        
                        // Add labels for high-confidence points
                        if (keypoint.score > 0.7) {
                            this.ctx.fillStyle = 'white';
                            this.ctx.font = '10px Arial';
                            this.ctx.fillText(this.keypointNames[index], keypoint.x + 5, keypoint.y - 5);
                        }
                    }
                });
            }
        });
    }
    
    /**
     * Calculate and update FPS counter
     */
    updatePerformanceStats() {
        this.frameCount++;
        const currentTime = Date.now();
        
        if (currentTime - this.lastTime >= 1000) {
            this.fps = Math.round((this.frameCount * 1000) / (currentTime - this.lastTime));
            this.frameCount = 0;
            this.lastTime = currentTime;
        }
    }
    
    /**
     * Update pose information display
     * @param {Array} poses - Detected pose data
     */
    updatePoseInfo(poses) {
        const poseCount = poses.length;
        let avgConfidence = 0;
        let keypointsInfo = '<strong>Keypoints Info:</strong><br>';
        
        if (poses.length > 0) {
            // Calculate average confidence across all poses
            const confidenceSum = poses.reduce((sum, pose) => {
                return sum + pose.keypoints.reduce((kpSum, kp) => kpSum + kp.score, 0) / pose.keypoints.length;
            }, 0);
            avgConfidence = Math.round((confidenceSum / poses.length) * 100);
            
            // Display details for first pose only
            const pose = poses[0];
            keypointsInfo += `<strong>Pose 1:</strong><br>`;
            pose.keypoints.forEach((kp, idx) => {
                if (kp.score > 0.5) {
                    keypointsInfo += `${this.keypointNames[idx]}: (${Math.round(kp.x)}, ${Math.round(kp.y)}) - ${Math.round(kp.score * 100)}%<br>`;
                }
            });
        } else {
            keypointsInfo += 'No poses detected';
        }
        
        this.updateStats(this.fps, poseCount, avgConfidence);
        document.getElementById('keypointsInfo').innerHTML = keypointsInfo;
    }
    
    /**
     * Update performance statistics display
     */
    updateStats(fps, poseCount, confidence) {
        document.getElementById('fpsValue').textContent = fps;
        document.getElementById('poseCount').textContent = poseCount;
        document.getElementById('confidenceValue').textContent = confidence + '%';
    }
    
    /**
     * Toggle skeleton visualization
     */
    toggleSkeleton() {
        this.showSkeleton = !this.showSkeleton;
        document.getElementById('toggleSkeleton').textContent = 
            this.showSkeleton ? '🦴 Hide Skeleton' : '🦴 Show Skeleton';
    }
    
    /**
     * Toggle keypoint visualization
     */
    toggleKeypoints() {
        this.showKeypoints = !this.showKeypoints;
        document.getElementById('toggleKeypoints').textContent = 
            this.showKeypoints ? '📍 Hide Points' : '📍 Show Points';
    }
    
    /**
     * Update status message
     */
    updateStatus(message) {
        document.getElementById('status').textContent = message;
    }
}

// Initialize detector when page loads
window.addEventListener('load', () => {
    new MoveNetPoseDetector();
});
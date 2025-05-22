# Guitar Finger Position Detector aka TabbyCat

This program uses MediaPipe and OpenCV to detect finger positions on a guitar fretboard and record them to a text file. It uses manual selection to define the fretboard area.

## Requirements

- Python 3.7+
- OpenCV
- MediaPipe
- NumPy

## How to Use

1. Run the script:
   ```
   python guitar_finger_detector.py
   ```

2. **Manual Fretboard Selection:**
   - When the program starts, you'll need to manually select the fretboard area
   - Click on the top-left corner of the fretboard in the camera view
   - Click on the bottom-right corner of the fretboard in the camera view

3. After the fretboard is selected, the program will:
   - Draw string lines and label them with their names (E, B, G, D, A, E)
   - Draw fret lines at the detected metal fret positions
   - Detect and track your fingers using MediaPipe's hand tracking
   - Identify which string and fret each finger is pressing
   - Display this information on screen
   - Write the detected positions to `guitar_positions.txt`

4. Controls:
   - Press 'm' to re-select the fretboard area
   - Press 'ESC' to exit the program

## Tips for Best Results

- Ensure good lighting on your guitar fretboard
- Position your camera so it has a clear view of the entire fretboard
- Keep your fingertips clearly visible to the camera
- Make sure to select the fretboard area accurately for best results

## How It Works

The program uses manual selection to define the fretboard area, then:
- Creates a representation of fret spacing (which is non-linear on real guitars)
- Maps finger positions to the actual fret locations
- Uses MediaPipe for accurate hand and finger tracking

## Output Format

The program writes to a file called `guitar_positions.txt` with content like:

## Inspiration/Research Material
https://www.idmil.org/project/guitar-fingering-recognition/
https://github.com/paulden/guitar-fingering-recognition

## Contact
samuel.keke at Outlook.com

// @ts-ignore
import express, {NextFunction, Request, Response} from 'express';
import * as faceapi from "face-api.js";
import {tensor3d} from "@tensorflow/tfjs-core";
require('@tensorflow/tfjs-node');
const router = express.Router();


// getting the emotions
const getEmotions = async (req: Request, res: Response, next: NextFunction) => {
    await faceapi.nets.tinyFaceDetector.loadFromDisk('weights')
    await faceapi.nets.faceLandmark68Net.loadFromDisk('weights')
    await faceapi.nets.faceRecognitionNet.loadFromDisk('weights')
    await faceapi.nets.faceExpressionNet.loadFromDisk('weights')

    let image = req.body.image;
    let tensor = tensor3d(image, [540, 960, 3])

    const results = await faceapi.detectAllFaces(tensor,
      new faceapi.TinyFaceDetectorOptions())
    .withFaceLandmarks()
    .withFaceExpressions()

    const displaySize = { width: 960, height: 540 }
    const resizedDetections = faceapi.resizeResults(results, displaySize)
    if (resizedDetections[0] != undefined) {
        return res.status(200).json({
            message: resizedDetections[0]["expressions"]
        });
    } else {
        return res.status(200).json({
            message: "undefined"
        });
    }

};

router.post('/emotions', getEmotions);

export = router;

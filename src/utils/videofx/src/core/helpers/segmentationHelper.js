"use strict";
exports.__esModule = true;
exports.getTFLiteModelFileName = exports.inputResolutions = void 0;
exports.inputResolutions = {
    '640x360': [640, 360],
    '256x256': [256, 256],
    '256x144': [256, 144],
    '160x96': [160, 96]
};
function getTFLiteModelFileName(model, inputResolution) {
    switch (model) {
        case 'meet':
            return inputResolution === '256x144' ? 'segm_full_v679' : 'segm_lite_v681';
        case 'mlkit':
            return 'selfiesegmentation_mlkit-256x256-2021_01_19-v1215.f16';
        default:
            throw new Error("No TFLite file for this segmentation model: " + model);
    }
}
exports.getTFLiteModelFileName = getTFLiteModelFileName;

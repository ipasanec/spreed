import createTFLiteModule from '../virtual-background/vendor/tflite/tflite'
import createTFLiteSIMDModule from '../virtual-background/vendor/tflite/tflite-simd'

const models = {
	model96: 'libs/segm_lite_v681.tflite',
	model144: 'libs/segm_full_v679.tflite',
}

const segmentationDimensions = {
	model96: {
		height: 96,
		width: 160,
	},
	model144: {
		height: 144,
		width: 256,
	},
}

self.addEventListener('message', (e) => {
	const data = e.data
	makeTFLite(data)
})

async function makeTFLite(message) {
	switch (message) {
	case 'simd':
		// TODO: fetch and initiate simd.
		self.tflite = await createTFLiteSIMDModule()
		break
	case 'wasm':
		// TODO: fetch and initiate wasm without simd.
		self.tflite = await createTFLiteModule()
		break
	default:
		self.postMessage(false)
		return
	}
	self.modelBufferOffset = self.tflite._getModelBufferMemoryOffset()
	// TODO: Figure out wasm check.
	self.modelResponse = await fetch(wasmCheck.feature.simd ? models.model144 : models.model96)

	if (!self.modelResponse.ok) {
		throw new Error('Failed to download tflite model!')
	}
	self.model = await self.modelResponse.arrayBuffer()

	self.tflite.HEAPU8.set(new Uint8Array(self.model), self.modelBufferOffset)

	self.tflite._loadModel(self.model.byteLength)

	self.postMessage(self.tflite)

	// TODO: Needs to be done in the global scope.
	// const options = {
	// 	...wasmCheck.feature.simd ? segmentationDimensions.model144 : segmentationDimensions.model96,
	// 	virtualBackground,
	// }
}

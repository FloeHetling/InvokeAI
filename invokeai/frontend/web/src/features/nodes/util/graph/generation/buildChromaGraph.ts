import { logger } from 'app/logging/logger';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { selectMainModelConfig, selectParamsSlice } from 'features/controlLayers/store/paramsSlice';
import { selectRefImagesSlice } from 'features/controlLayers/store/refImagesSlice';
import { selectCanvasMetadata } from 'features/controlLayers/store/selectors';
import { addFLUXReduxes } from 'features/nodes/util/graph/generation/addFLUXRedux';
import { addImageToImage } from 'features/nodes/util/graph/generation/addImageToImage';
import { addInpaint } from 'features/nodes/util/graph/generation/addInpaint';
import { addNSFWChecker } from 'features/nodes/util/graph/generation/addNSFWChecker';
import { addOutpaint } from 'features/nodes/util/graph/generation/addOutpaint';
import { addTextToImage } from 'features/nodes/util/graph/generation/addTextToImage';
import { addWatermarker } from 'features/nodes/util/graph/generation/addWatermarker';
import { Graph } from 'features/nodes/util/graph/generation/Graph';
import { selectCanvasOutputFields } from 'features/nodes/util/graph/graphBuilderUtils';
import type { GraphBuilderArg, GraphBuilderReturn, ImageOutputNodes } from 'features/nodes/util/graph/types';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import type { Invocation } from 'services/api/types';
import { isSelfContainedChromaPipeline } from 'services/api/types';
import type { Equals } from 'tsafe';
import { assert } from 'tsafe';

const log = logger('system');

export const buildChromaGraph = async (arg: GraphBuilderArg): Promise<GraphBuilderReturn> => {
  const { generationMode, state, manager } = arg;

  log.debug({ generationMode, manager: manager?.id }, 'Building Chroma graph');

  const model = selectMainModelConfig(state);
  assert(model, 'No model selected');
  assert(model.base === 'chroma', 'Selected model is not a Chroma model');

  const params = selectParamsSlice(state);
  const refImages = selectRefImagesSlice(state);
  const { cfgScale: cfg_scale, steps, chromaScheduler: scheduler, fluxVAE, t5EncoderModel } = params;
  const mainSuppliesComponents = isSelfContainedChromaPipeline(model);

  if (!mainSuppliesComponents) {
    assert(t5EncoderModel, 'No T5 Encoder model found in state');
    assert(fluxVAE, 'No FLUX VAE model found in state');
  }

  const g = new Graph(getPrefixedId('chroma_graph'));
  const modelLoader = g.addNode({
    type: 'chroma_model_loader',
    id: getPrefixedId('chroma_model_loader'),
    model,
    ...(mainSuppliesComponents
      ? {}
      : {
          t5_encoder_model: t5EncoderModel!,
          vae_model: fluxVAE!,
        }),
  });

  const positivePrompt = g.addNode({
    id: getPrefixedId('positive_prompt'),
    type: 'string',
  });
  const negativePrompt = g.addNode({
    id: getPrefixedId('negative_prompt'),
    type: 'string',
  });
  const posCond = g.addNode({
    type: 'chroma_text_encoder',
    id: getPrefixedId('chroma_positive_text_encoder'),
  });
  const negCond = g.addNode({
    type: 'chroma_text_encoder',
    id: getPrefixedId('chroma_negative_text_encoder'),
  });
  const seed = g.addNode({
    id: getPrefixedId('seed'),
    type: 'integer',
  });
  const denoise = g.addNode({
    type: 'chroma_denoise',
    id: getPrefixedId('chroma_denoise'),
    cfg_scale,
    num_steps: steps,
    scheduler,
  });
  const l2i = g.addNode({
    type: 'flux_vae_decode',
    id: getPrefixedId('flux_vae_decode'),
  });

  g.addEdge(modelLoader, 'transformer', denoise, 'transformer');
  g.addEdge(modelLoader, 't5_encoder', posCond, 't5_encoder');
  g.addEdge(modelLoader, 't5_encoder', negCond, 't5_encoder');
  g.addEdge(modelLoader, 'vae', l2i, 'vae');
  g.addEdge(positivePrompt, 'value', posCond, 'prompt');
  g.addEdge(negativePrompt, 'value', negCond, 'prompt');
  g.addEdge(posCond, 'conditioning', denoise, 'positive_text_conditioning');
  g.addEdge(negCond, 'conditioning', denoise, 'negative_text_conditioning');
  g.addEdge(seed, 'value', denoise, 'seed');
  g.addEdge(denoise, 'latents', l2i, 'latents');

  g.upsertMetadata({
    cfg_scale,
    model: Graph.getModelMetadataField(model),
    scheduler,
    steps,
    t5_encoder: mainSuppliesComponents ? undefined : t5EncoderModel,
    vae: mainSuppliesComponents ? undefined : fluxVAE,
  });
  g.addEdgeToMetadata(seed, 'value', 'seed');
  g.addEdgeToMetadata(positivePrompt, 'value', 'positive_prompt');
  g.addEdgeToMetadata(negativePrompt, 'value', 'negative_prompt');

  const fluxReduxCollect = g.addNode({
    type: 'collect',
    id: getPrefixedId('flux_redux_collector'),
  });
  const { addedFLUXReduxes } = addFLUXReduxes({
    entities: refImages.entities,
    g,
    collector: fluxReduxCollect,
    model,
  });
  if (addedFLUXReduxes > 0) {
    g.addEdge(fluxReduxCollect, 'collection', denoise, 'redux_conditioning');
  } else {
    g.deleteNode(fluxReduxCollect.id);
  }

  let canvasOutput: Invocation<ImageOutputNodes> = l2i;

  if (generationMode === 'txt2img') {
    canvasOutput = addTextToImage({ g, state, denoise, l2i });
    g.upsertMetadata({ generation_mode: 'chroma_txt2img' });
  } else if (generationMode === 'img2img') {
    assert(manager !== null);
    const i2l = g.addNode({
      type: 'flux_vae_encode',
      id: getPrefixedId('flux_vae_encode'),
    });
    canvasOutput = await addImageToImage({ g, state, manager, denoise, l2i, i2l, vaeSource: modelLoader });
    g.upsertMetadata({ generation_mode: 'chroma_img2img' });
  } else if (generationMode === 'inpaint') {
    assert(manager !== null);
    const i2l = g.addNode({
      type: 'flux_vae_encode',
      id: getPrefixedId('flux_vae_encode'),
    });
    canvasOutput = await addInpaint({
      g,
      state,
      manager,
      denoise,
      l2i,
      i2l,
      vaeSource: modelLoader,
      modelLoader,
      seed,
    });
    g.upsertMetadata({ generation_mode: 'chroma_inpaint' });
  } else if (generationMode === 'outpaint') {
    assert(manager !== null);
    const i2l = g.addNode({
      type: 'flux_vae_encode',
      id: getPrefixedId('flux_vae_encode'),
    });
    canvasOutput = await addOutpaint({
      g,
      state,
      manager,
      denoise,
      l2i,
      i2l,
      vaeSource: modelLoader,
      modelLoader,
      seed,
    });
    g.upsertMetadata({ generation_mode: 'chroma_outpaint' });
  } else {
    assert<Equals<typeof generationMode, never>>(false);
  }

  if (state.system.shouldUseNSFWChecker) {
    canvasOutput = addNSFWChecker(g, canvasOutput);
  }
  if (state.system.shouldUseWatermarker) {
    canvasOutput = addWatermarker(g, canvasOutput);
  }

  g.updateNode(canvasOutput, selectCanvasOutputFields(state));
  if (selectActiveTab(state) === 'canvas') {
    g.upsertMetadata(selectCanvasMetadata(state));
  }
  g.setMetadataReceivingNode(canvasOutput);

  return { g, seed, positivePrompt, negativePrompt };
};

import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('app/logging/logger', () => ({
  logger: () => ({ debug: vi.fn() }),
}));

let nextId = 0;
vi.mock('features/controlLayers/konva/util', () => ({
  getPrefixedId: (prefix: string) => `${prefix}:${nextId++}`,
}));

const chromaCheckpoint = {
  key: 'chroma-checkpoint',
  hash: 'chroma-checkpoint-hash',
  name: 'Chroma1-HD',
  base: 'chroma',
  type: 'main',
  format: 'checkpoint',
};

const chromaDiffusersPipeline = {
  key: 'chroma-diffusers',
  hash: 'chroma-diffusers-hash',
  name: 'Chroma1-HD Diffusers',
  base: 'chroma',
  type: 'main',
  format: 'diffusers',
  submodels: {
    transformer: {},
    vae: {},
    text_encoder: {},
    tokenizer: {},
  },
};

const t5EncoderModel = {
  key: 't5-xxl',
  hash: 't5-xxl-hash',
  name: 'T5 XXL FP8',
  base: 'any',
  type: 't5_encoder',
};

const fluxVAE = {
  key: 'flux-vae',
  hash: 'flux-vae-hash',
  name: 'FLUX VAE',
  base: 'flux',
  type: 'vae',
};

let currentModel: Record<string, unknown> = chromaCheckpoint;
let currentParams: Record<string, unknown> = {};
let currentRefImages: { entities: unknown[]; selectedEntityId: string | null; isPanelOpen: boolean } = {
  entities: [],
  selectedEntityId: null,
  isPanelOpen: false,
};

vi.mock('features/controlLayers/store/paramsSlice', () => ({
  selectMainModelConfig: vi.fn(() => currentModel),
  selectParamsSlice: vi.fn(() => currentParams),
}));

vi.mock('features/controlLayers/store/refImagesSlice', () => ({
  selectRefImagesSlice: vi.fn(() => currentRefImages),
}));

vi.mock('features/controlLayers/store/selectors', () => ({
  selectCanvasMetadata: vi.fn(() => ({})),
}));

vi.mock('features/ui/store/uiSelectors', () => ({
  selectActiveTab: vi.fn(() => 'generate'),
}));

vi.mock('features/nodes/util/graph/graphBuilderUtils', () => ({
  selectCanvasOutputFields: vi.fn(() => ({})),
}));

vi.mock('features/nodes/util/graph/generation/addTextToImage', () => ({
  addTextToImage: vi.fn(({ l2i }) => l2i),
}));
vi.mock('features/nodes/util/graph/generation/addImageToImage', () => ({ addImageToImage: vi.fn() }));
vi.mock('features/nodes/util/graph/generation/addInpaint', () => ({ addInpaint: vi.fn() }));
vi.mock('features/nodes/util/graph/generation/addOutpaint', () => ({ addOutpaint: vi.fn() }));
vi.mock('features/nodes/util/graph/generation/addNSFWChecker', () => ({
  addNSFWChecker: vi.fn((_g, node) => node),
}));
vi.mock('features/nodes/util/graph/generation/addWatermarker', () => ({
  addWatermarker: vi.fn((_g, node) => node),
}));

import type { GraphBuilderArg } from 'features/nodes/util/graph/types';

import { buildChromaGraph } from './buildChromaGraph';

const buildGraphArg = (): GraphBuilderArg =>
  ({
    generationMode: 'txt2img',
    manager: null,
    state: {
      system: {
        shouldUseNSFWChecker: false,
        shouldUseWatermarker: false,
      },
    },
  }) as unknown as GraphBuilderArg;

const findNode = (nodes: Record<string, { type: string }>, type: string) =>
  Object.values(nodes).find((node) => node.type === type) as Record<string, unknown> | undefined;

beforeEach(() => {
  nextId = 0;
  currentModel = chromaCheckpoint;
  currentRefImages = { entities: [], selectedEntityId: null, isPanelOpen: false };
  currentParams = {
    cfgScale: 2.5,
    steps: 25,
    chromaScheduler: 'euler',
    fluxVAE,
    t5EncoderModel,
  };
});

describe('buildChromaGraph', () => {
  it('builds a native T5-only Chroma graph for a single-file checkpoint', async () => {
    const { g } = await buildChromaGraph(buildGraphArg());
    const graph = g.getGraph();
    const loader = findNode(graph.nodes, 'chroma_model_loader');
    const denoise = findNode(graph.nodes, 'chroma_denoise');
    const metadata = findNode(graph.nodes, 'core_metadata');

    expect(loader).toEqual(
      expect.objectContaining({
        model: chromaCheckpoint,
        t5_encoder_model: t5EncoderModel,
        vae_model: fluxVAE,
      })
    );
    expect(denoise).toEqual(expect.objectContaining({ cfg_scale: 2.5, num_steps: 25, scheduler: 'euler' }));
    expect(Object.values(graph.nodes).filter((node) => node.type === 'chroma_text_encoder')).toHaveLength(2);
    expect(Object.values(graph.nodes).some((node) => node.type === 'flux_text_encoder')).toBe(false);
    expect(Object.values(graph.nodes).some((node) => node.type === 'flux_vae_decode')).toBe(true);
    expect(metadata).toEqual(
      expect.objectContaining({
        generation_mode: 'chroma_txt2img',
        t5_encoder: t5EncoderModel,
        vae: fluxVAE,
      })
    );
  });

  it('uses components bundled in a complete Diffusers pipeline', async () => {
    currentModel = chromaDiffusersPipeline;
    currentParams = {
      cfgScale: 3,
      steps: 40,
      chromaScheduler: 'heun',
      fluxVAE: null,
      t5EncoderModel: null,
    };

    const { g } = await buildChromaGraph(buildGraphArg());
    const graph = g.getGraph();
    const loader = findNode(graph.nodes, 'chroma_model_loader');
    const metadata = findNode(graph.nodes, 'core_metadata');

    expect(loader).toEqual(expect.objectContaining({ model: chromaDiffusersPipeline }));
    expect(loader?.t5_encoder_model).toBeUndefined();
    expect(loader?.vae_model).toBeUndefined();
    expect(metadata?.t5_encoder).toBeUndefined();
    expect(metadata?.vae).toBeUndefined();
  });

  it('passes the Chroma-only Euler CFG++ Beta scheduler to the denoise node', async () => {
    currentParams = {
      cfgScale: 2.5,
      steps: 25,
      chromaScheduler: 'euler_cfg_pp_beta',
      fluxVAE,
      t5EncoderModel,
    };

    const { g } = await buildChromaGraph(buildGraphArg());
    const graph = g.getGraph();
    const denoise = findNode(graph.nodes, 'chroma_denoise');

    expect(denoise).toEqual(expect.objectContaining({ scheduler: 'euler_cfg_pp_beta' }));
  });

  it('rejects a single-file checkpoint when standalone components are not selected', async () => {
    currentParams = {
      cfgScale: 2.5,
      steps: 25,
      chromaScheduler: 'euler',
      fluxVAE: null,
      t5EncoderModel: null,
    };

    await expect(buildChromaGraph(buildGraphArg())).rejects.toThrow(/T5 Encoder/);
  });

  it('wires FLUX Redux reference images into Chroma denoise', async () => {
    currentRefImages = {
      selectedEntityId: null,
      isPanelOpen: false,
      entities: [
        {
          id: 'redux-reference',
          isEnabled: true,
          config: {
            type: 'flux_redux',
            model: { key: 'flux-redux', name: 'FLUX Redux', base: 'flux', type: 'flux_redux' },
            image: {
              original: { image: { image_name: 'reference.png', width: 1024, height: 1024 } },
            },
            downsamplingFactor: 7,
            weight: 0.15,
          },
        },
      ],
    };

    const { g } = await buildChromaGraph(buildGraphArg());
    const graph = g.getGraph();
    const redux = findNode(graph.nodes, 'flux_redux');
    const collector = findNode(graph.nodes, 'collect');
    const denoise = findNode(graph.nodes, 'chroma_denoise');
    const metadata = findNode(graph.nodes, 'core_metadata');

    expect(redux).toEqual(
      expect.objectContaining({
        redux_model: expect.objectContaining({ key: 'flux-redux', type: 'flux_redux' }),
        image: { image_name: 'reference.png' },
        downsampling_factor: 7,
        weight: 0.15,
      })
    );
    expect(collector).toBeDefined();
    expect(denoise).toBeDefined();
    expect(g.getEdges()).toEqual(
      expect.arrayContaining([
        {
          source: { node_id: redux?.id, field: 'redux_cond' },
          destination: { node_id: collector?.id, field: 'item' },
        },
        {
          source: { node_id: collector?.id, field: 'collection' },
          destination: { node_id: denoise?.id, field: 'redux_conditioning' },
        },
      ])
    );
    expect(metadata?.ref_images).toEqual(currentRefImages.entities);
  });
});

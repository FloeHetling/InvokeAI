import type { FLUXReduxConfig, FLUXReduxImageInfluence, RefImageState } from 'features/controlLayers/store/types';
import { isFLUXReduxConfig } from 'features/controlLayers/store/types';
import { getGlobalReferenceImageWarnings } from 'features/controlLayers/store/validators';
import type { Graph } from 'features/nodes/util/graph/generation/Graph';
import type { Invocation, MainModelConfig } from 'services/api/types';
import { assert } from 'tsafe';

type AddFLUXReduxResult = {
  addedFLUXReduxes: number;
};

type AddFLUXReduxArg = {
  entities: RefImageState[];
  g: Graph;
  collector: Invocation<'collect'>;
  model: MainModelConfig;
};

export const addFLUXReduxes = ({ entities, g, collector, model }: AddFLUXReduxArg): AddFLUXReduxResult => {
  const validFLUXReduxes = entities
    .filter((entity) => entity.isEnabled)
    .filter((entity) => isFLUXReduxConfig(entity.config))
    .filter((entity) => getGlobalReferenceImageWarnings(entity, model).length === 0);

  const result: AddFLUXReduxResult = {
    addedFLUXReduxes: 0,
  };

  for (const { id, config } of validFLUXReduxes) {
    assert(isFLUXReduxConfig(config), 'This should have been filtered out');
    result.addedFLUXReduxes++;

    addFLUXRedux(id, config, g, collector);
  }

  g.upsertMetadata({ ref_images: validFLUXReduxes }, 'merge');

  return result;
};

/**
 * Preset mapping used by regional Redux. Global Redux reference images store raw
 * downsampling factor and weight so users can tune both values independently.
 */
export const IMAGE_INFLUENCE_TO_SETTINGS: Record<
  FLUXReduxImageInfluence,
  Pick<Invocation<'flux_redux'>, 'downsampling_factor' | 'downsampling_function' | 'weight'>
> = {
  lowest: {
    downsampling_factor: 5,
    // downsampling_function: 'area',
    weight: 1,
  },
  low: {
    downsampling_factor: 4,
    // downsampling_function: 'area',
    weight: 1,
  },
  medium: {
    downsampling_factor: 3,
    // downsampling_function: 'area',
    weight: 1,
  },
  high: {
    downsampling_factor: 2,
    // downsampling_function: 'area',
    weight: 1,
  },
  highest: {
    downsampling_factor: 1,
    // downsampling_function: 'area',
    weight: 1,
  },
};

const addFLUXRedux = (id: string, ipAdapter: FLUXReduxConfig, g: Graph, collector: Invocation<'collect'>) => {
  const { model: fluxReduxModel, image } = ipAdapter;
  assert(image, 'FLUX Redux image is required');
  assert(fluxReduxModel, 'FLUX Redux model is required');

  const node = g.addNode({
    id: `flux_redux_${id}`,
    type: 'flux_redux',
    redux_model: fluxReduxModel,
    image: {
      image_name: image.crop?.image.image_name ?? image.original.image.image_name,
    },
    downsampling_factor: ipAdapter.downsamplingFactor,
    weight: ipAdapter.weight,
  });

  g.addEdge(node, 'redux_cond', collector, 'item');
};

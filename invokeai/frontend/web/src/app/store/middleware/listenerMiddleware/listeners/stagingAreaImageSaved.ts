import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { stagingAreaImageSaved } from 'features/canvas/store/actions';
import { toast } from 'features/toast/toast';
import { t } from 'i18next';
import { imagesApi } from 'services/api/endpoints/images';

export const addStagingAreaImageSavedListener = (startAppListening: AppStartListening) => {
  startAppListening({
    actionCreator: stagingAreaImageSaved,
    effect: async (action, { dispatch, getState }) => {
      const { imageDTO } = action.payload;

      try {
        const newImageDTO = await dispatch(
          imagesApi.endpoints.changeImageIsIntermediate.initiate({
            imageDTO,
            is_intermediate: false,
          })
        ).unwrap();

        // we may need to add it to the autoadd board
        const { autoAddBoardId } = getState().gallery;

        if (autoAddBoardId && autoAddBoardId !== 'none') {
          await dispatch(
            imagesApi.endpoints.addImageToBoard.initiate({
              imageDTO: newImageDTO,
              board_id: autoAddBoardId,
            })
          );
        }
        toast({ id: 'IMAGE_SAVED', title: t('toast.imageSaved'), status: 'success' });
      } catch (error) {
        toast({
          id: 'IMAGE_SAVE_FAILED',
          title: t('toast.imageSavingFailed'),
          description: (error as Error)?.message,
          status: 'error',
        });
      }
    },
  });
};

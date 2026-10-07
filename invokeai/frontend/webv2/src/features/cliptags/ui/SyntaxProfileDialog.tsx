import type { ClipSyntaxProfile, ClipSyntaxProfileOptions } from '@features/cliptags/core/types';

import { Dialog, Input, Portal, Stack, Switch, Text } from '@chakra-ui/react';
import { createClipSyntaxProfile, updateClipSyntaxProfile } from '@features/cliptags/data/api';
import { useExitRetainedValue } from '@platform/react/useExitRetainedValue';
import { getApiErrorMessage } from '@platform/transport/http';
import { Button, CloseButton } from '@platform/ui/Button';
import { Field } from '@platform/ui/Field';
import { useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { notify, useClipTagWrite } from './clipTagWrites';
import { ErrorAlert } from './ErrorAlert';

/** What a new profile starts with; the backend's own defaults. */
export const DEFAULT_SYNTAX_PROFILE_OPTIONS: ClipSyntaxProfileOptions = {
  appendTypeParentheses: false,
  escapeColons: true,
  escapeParentheses: true,
  prefixArtistWithBy: false,
  spacesToUnderscores: true,
};

export const SYNTAX_PROFILE_OPTIONS: readonly { labelKey: string; option: keyof ClipSyntaxProfileOptions }[] = [
  { labelKey: 'cliptags.manager.syntaxProfiles.options.spacesToUnderscores', option: 'spacesToUnderscores' },
  { labelKey: 'cliptags.manager.syntaxProfiles.options.escapeParentheses', option: 'escapeParentheses' },
  { labelKey: 'cliptags.manager.syntaxProfiles.options.escapeColons', option: 'escapeColons' },
  { labelKey: 'cliptags.manager.syntaxProfiles.options.appendTypeParentheses', option: 'appendTypeParentheses' },
  { labelKey: 'cliptags.manager.syntaxProfiles.options.prefixArtistWithBy', option: 'prefixArtistWithBy' },
];

export type SyntaxProfileDialogTarget = { mode: 'create' } | { mode: 'edit'; profile: ClipSyntaxProfile };

const SWITCH_CHECKED_STYLES = { bg: 'accent.solid' };

/** Only the rendering options: a profile also carries its `id` and `name`, which must not travel with them. */
const getProfileOptions = (profile: ClipSyntaxProfileOptions): ClipSyntaxProfileOptions => ({
  appendTypeParentheses: profile.appendTypeParentheses,
  escapeColons: profile.escapeColons,
  escapeParentheses: profile.escapeParentheses,
  prefixArtistWithBy: profile.prefixArtistWithBy,
  spacesToUnderscores: profile.spacesToUnderscores,
});

/** Creates a syntax profile, or edits a profile's name and its five rendering options. */
export const SyntaxProfileDialog = ({
  onClose,
  target,
}: {
  target: SyntaxProfileDialogTarget | null;
  onClose: () => void;
}) => {
  const { t } = useTranslation();
  const save = useClipTagWrite(
    ({
      name,
      options,
      target: saved,
    }: {
      name: string;
      options: ClipSyntaxProfileOptions;
      target: SyntaxProfileDialogTarget;
    }) =>
      saved.mode === 'edit'
        ? updateClipSyntaxProfile(saved.profile.id, { ...options, name })
        : createClipSyntaxProfile(name, options)
  );
  const { isPending, reset } = save;
  const shown = useExitRetainedValue(target);
  const { release } = shown;

  const handleOpenChange = useCallback(
    (event: { open: boolean }) => {
      if (!event.open && !isPending) {
        onClose();
      }
    },
    [isPending, onClose]
  );
  const handleExitComplete = useCallback(() => {
    release();
    reset();
  }, [release, reset]);
  const handleSave = useCallback(
    async (name: string, options: ClipSyntaxProfileOptions) => {
      if (!shown.value) {
        return;
      }

      const isEdit = shown.value.mode === 'edit';

      try {
        await save.mutateAsync({ name, options, target: shown.value });
        notify(
          'success',
          isEdit ? t('cliptags.manager.syntaxProfiles.updated') : t('cliptags.manager.syntaxProfiles.created')
        );
        onClose();
      } catch {
        // The dialog shows the failure from the mutation's state.
      }
    },
    [onClose, save, shown.value, t]
  );

  return (
    <Dialog.Root
      closeOnEscape={!isPending}
      closeOnInteractOutside={!isPending}
      lazyMount
      open={target !== null}
      scrollBehavior="inside"
      size="sm"
      unmountOnExit
      onExitComplete={handleExitComplete}
      onOpenChange={handleOpenChange}
    >
      <Portal>
        <Dialog.Backdrop />
        <Dialog.Positioner>
          <Dialog.Content>
            {shown.value ? (
              <SyntaxProfileForm
                key={shown.generation}
                error={
                  save.isError
                    ? getApiErrorMessage(
                        save.error,
                        shown.value.mode === 'edit'
                          ? t('cliptags.manager.syntaxProfiles.updateFailed')
                          : t('cliptags.manager.syntaxProfiles.createFailed')
                      )
                    : null
                }
                isPending={isPending}
                target={shown.value}
                onClose={onClose}
                onSave={handleSave}
              />
            ) : null}
            <Dialog.CloseTrigger asChild>
              <CloseButton disabled={isPending} />
            </Dialog.CloseTrigger>
          </Dialog.Content>
        </Dialog.Positioner>
      </Portal>
    </Dialog.Root>
  );
};

const SyntaxProfileForm = ({
  error,
  isPending,
  onClose,
  onSave,
  target,
}: {
  error: string | null;
  isPending: boolean;
  target: SyntaxProfileDialogTarget;
  onClose: () => void;
  onSave: (name: string, options: ClipSyntaxProfileOptions) => void;
}) => {
  const { t } = useTranslation();
  const isEdit = target.mode === 'edit';
  const [name, setName] = useState(isEdit ? target.profile.name : '');
  const [options, setOptions] = useState<ClipSyntaxProfileOptions>(
    isEdit ? getProfileOptions(target.profile) : DEFAULT_SYNTAX_PROFILE_OPTIONS
  );
  const isNameValid = name.trim() !== '';

  const handleNameChange = useCallback((event: { currentTarget: { value: string } }) => {
    setName(event.currentTarget.value);
  }, []);
  const handleSubmit = useCallback(
    (event: { preventDefault: () => void }) => {
      event.preventDefault();

      if (isNameValid && !isPending) {
        onSave(name.trim(), options);
      }
    },
    [isNameValid, isPending, name, onSave, options]
  );

  return (
    <form onSubmit={handleSubmit}>
      <Dialog.Header borderBottomWidth="1px" borderColor="border.subtle">
        <Dialog.Title>
          {isEdit ? t('cliptags.manager.syntaxProfiles.editTitle') : t('cliptags.manager.syntaxProfiles.newTitle')}
        </Dialog.Title>
      </Dialog.Header>
      <Dialog.Body>
        <Stack gap="4">
          <Field disabled={isPending} label={t('common.name')}>
            <Input
              autoComplete="off"
              placeholder={t('cliptags.manager.syntaxProfiles.namePlaceholder')}
              value={name}
              onChange={handleNameChange}
            />
          </Field>
          <Stack gap="3">
            <Text color="fg.muted" fontSize="xs" fontWeight="600" textTransform="uppercase">
              {t('cliptags.manager.syntaxProfiles.options.heading')}
            </Text>
            {SYNTAX_PROFILE_OPTIONS.map(({ labelKey, option }) => (
              <OptionSwitch
                key={option}
                checked={options[option]}
                disabled={isPending}
                label={t(labelKey)}
                option={option}
                onChange={setOptions}
              />
            ))}
          </Stack>
          {error ? <ErrorAlert message={error} /> : null}
        </Stack>
      </Dialog.Body>
      <Dialog.Footer>
        <Button disabled={isPending} variant="ghost" onClick={onClose}>
          {t('common.cancel')}
        </Button>
        <Button disabled={!isNameValid || isPending} loading={isPending} type="submit" variant="solid">
          {isEdit ? t('common.save') : t('cliptags.manager.syntaxProfiles.create')}
        </Button>
      </Dialog.Footer>
    </form>
  );
};

const OptionSwitch = ({
  checked,
  disabled,
  label,
  onChange,
  option,
}: {
  checked: boolean;
  disabled: boolean;
  label: string;
  option: keyof ClipSyntaxProfileOptions;
  onChange: (update: (current: ClipSyntaxProfileOptions) => ClipSyntaxProfileOptions) => void;
}) => {
  const handleCheckedChange = useCallback(
    (event: { checked: boolean }) => onChange((current) => ({ ...current, [option]: event.checked })),
    [onChange, option]
  );

  return (
    <Switch.Root
      alignItems="center"
      checked={checked}
      disabled={disabled}
      display="flex"
      justifyContent="space-between"
      w="full"
      onCheckedChange={handleCheckedChange}
    >
      <Switch.Label color="fg" fontSize="md" m="0">
        {label}
      </Switch.Label>
      <Switch.HiddenInput />
      <Switch.Control _checked={SWITCH_CHECKED_STYLES}>
        <Switch.Thumb />
      </Switch.Control>
    </Switch.Root>
  );
};

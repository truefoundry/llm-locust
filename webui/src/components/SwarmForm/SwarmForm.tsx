import { useState } from 'react';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import {
  Accordion,
  AccordionDetails,
  AccordionSummary,
  Alert,
  Box,
  Button,
  Container,
  Checkbox,
  FormControlLabel,
  SelectChangeEvent,
  TextField,
  Typography,
} from '@mui/material';
import { AlertColor } from '@mui/material/Alert';
import { connect } from 'react-redux';

import Form from 'components/Form/Form';
import NumericField from 'components/Form/NumericField';
import Select from 'components/Form/Select';
import CustomParameters from 'components/SwarmForm/SwarmCustomParameters';
import SwarmUserClassPicker from 'components/SwarmForm/SwarmUserClassPicker';
import { SWARM_STATE } from 'constants/swarm';
import { useStartSwarmMutation } from 'redux/api/swarm';
import { swarmActions } from 'redux/slice/swarm.slice';
import { IRootState } from 'redux/store';
import { ISwarmFormInput, ISwarmState } from 'types/swarm.types';
import { isEmpty } from 'utils/object';

interface IDispatchProps {
  setSwarm: (swarmPayload: Partial<ISwarmState>) => void;
}

export interface ISwarmFormProps {
  alert?: {
    level?: AlertColor;
    message: string;
  };
  isDisabled?: boolean;
  isEditSwarm?: boolean;
  onFormChange?: (formData: React.ChangeEvent<HTMLFormElement>) => void;
  onFormSubmit?: (inputData: ISwarmFormInput) => void;
}

interface ISwarmForm
  extends IDispatchProps,
    Pick<
      ISwarmState,
      | 'availableShapeClasses'
      | 'availableUserClasses'
      | 'extraOptions'
      | 'hideCommonOptions'
      | 'shapeUseCommonOptions'
      | 'host'
      | 'overrideHostWarning'
      | 'showUserclassPicker'
      | 'spawnRate'
      | 'numUsers'
      | 'userCount'
      | 'tokenizer'
      | 'model'
      | 'maxTokens'
      | 'promptMinTokens'
      | 'promptMaxTokens'
      | 'useRandomPrompts'
      | 'useSinglePrompt'
      | 'ignoreEos'
      | 'openaiApiKey'
    >,
    ISwarmFormProps {}

function SwarmForm({
  availableShapeClasses,
  availableUserClasses,
  host,
  extraOptions,
  hideCommonOptions,
  shapeUseCommonOptions,
  numUsers,
  userCount,
  overrideHostWarning,
  setSwarm,
  showUserclassPicker,
  spawnRate,
  alert,
  isDisabled = false,
  isEditSwarm = false,
  onFormChange,
  onFormSubmit,
  tokenizer,
  model,
  maxTokens,
  promptMinTokens,
  promptMaxTokens,
  useRandomPrompts,
  useSinglePrompt,
  ignoreEos,
  openaiApiKey,
}: ISwarmForm) {
  const [startSwarm] = useStartSwarmMutation();
  const [errorMessage, setErrorMessage] = useState('');
  const [selectedUserClasses, setSelectedUserClasses] = useState(availableUserClasses);
  const [promptMode, setPromptMode] = useState(() => {
    if (useRandomPrompts) {
      return {
        useRandomPrompts: true,
        useSinglePrompt: false,
        sampleFromDataset: false
      };
    } else if (useSinglePrompt !== false) {
      return {
        useRandomPrompts: false,
        useSinglePrompt: true,
        sampleFromDataset: false
      };
    } else {
      return {
        useRandomPrompts: false,
        useSinglePrompt: false,
        sampleFromDataset: true
      };
    }
  });

  const onStartSwarm = async (inputData: ISwarmFormInput) => {
    const { data } = await startSwarm({
      ...inputData,
      ...(showUserclassPicker && selectedUserClasses ? { userClasses: selectedUserClasses } : {}),
      useRandomPrompts: promptMode.useRandomPrompts,
      useSinglePrompt: promptMode.useSinglePrompt,
    });

    if (data && data.success) {
      setSwarm({
        state: SWARM_STATE.RUNNING,
        host: inputData.host || host,
        spawnRate: inputData.spawnRate,
        userCount: inputData.userCount,
        tokenizer: inputData.tokenizer,
        model: inputData.model,
        maxTokens: inputData.maxTokens,
        promptMinTokens: inputData.promptMinTokens,
        promptMaxTokens: inputData.promptMaxTokens,
        useRandomPrompts: promptMode.useRandomPrompts,
        useSinglePrompt: promptMode.useSinglePrompt,
        ignoreEos: inputData.ignoreEos,
        openaiApiKey: inputData.openaiApiKey,
      });
    } else {
      setErrorMessage(data ? data.message : 'An unknown error occured.');
    }

    if (onFormSubmit) {
      onFormSubmit({
        ...inputData,
        useRandomPrompts: promptMode.useRandomPrompts,
        useSinglePrompt: promptMode.useSinglePrompt,
      });
    }
  };

  const handleSwarmFormChange = (formEvent: React.ChangeEvent<HTMLFormElement>) => {
    if (errorMessage) {
      setErrorMessage('');
    }

    if (onFormChange) {
      onFormChange(formEvent);
    }
  };

  const onShapeClassChange = (event: SelectChangeEvent<unknown>) => {
    if (!shapeUseCommonOptions) {
      const hasSelectedShapeClass = event.target.value !== availableShapeClasses[0];
      setSwarm({
        hideCommonOptions: hasSelectedShapeClass,
      });
    }
  };

  const handlePromptModeChange = (mode: 'useRandomPrompts' | 'useSinglePrompt' | 'sampleFromDataset') => {
    if (mode === 'useRandomPrompts') {
      setPromptMode({
        useRandomPrompts: true,
        useSinglePrompt: false,
        sampleFromDataset: false,
      });
    } else if (mode === 'useSinglePrompt') {
      setPromptMode({
        useRandomPrompts: false,
        useSinglePrompt: true,
        sampleFromDataset: false,
      });
    } else {
      setPromptMode({
        useRandomPrompts: false,
        useSinglePrompt: false,
        sampleFromDataset: true,
      });
    }
  };

  return (
    <Container maxWidth='md' sx={{ my: 2 }}>
      <Typography component='h2' noWrap variant='h6'>
        {isEditSwarm ? 'Edit running load test' : 'Start new load test'}
      </Typography>
      {!isEditSwarm && showUserclassPicker && (
        <Box marginBottom={2} marginTop={2}>
          <SwarmUserClassPicker
            availableUserClasses={availableUserClasses}
            selectedUserClasses={selectedUserClasses}
            setSelectedUserClasses={setSelectedUserClasses}
          />
        </Box>
      )}
      <Form<ISwarmFormInput> onChange={handleSwarmFormChange} onSubmit={onStartSwarm}>
        <Box
          sx={{
            marginBottom: 2,
            marginTop: 2,
            display: 'flex',
            flexDirection: 'column',
            rowGap: 4,
          }}
        >
          {!isEditSwarm && showUserclassPicker && (
            <Select
              label='Shape Class'
              name='shapeClass'
              onChange={onShapeClassChange}
              options={availableShapeClasses}
            />
          )}
          <NumericField
            defaultValue={(hideCommonOptions && '0') || userCount || numUsers || 1}
            disabled={!!hideCommonOptions}
            label='Number of users (peak concurrency)'
            name='userCount'
            required
            title={hideCommonOptions ? 'Disabled for tests using LoadTestShape class' : ''}
          />
          <NumericField
            defaultValue={(hideCommonOptions && '0') || spawnRate || 1}
            disabled={!!hideCommonOptions}
            label='Ramp up (users started/second)'
            name='spawnRate'
            required
            title={hideCommonOptions ? 'Disabled for tests using LoadTestShape class' : ''}
          />
          {!isEditSwarm && (
            <>
              <TextField
                defaultValue={host}
                label={`Host  (OpenAI Compatible chat/completions endpoint) ${
                  overrideHostWarning
                    ? '(setting this will override the host for the User classes)'
                    : ''
                }`}
                name='host'
              />
              <TextField
                defaultValue={model || "nousresearch-meta-llama-3-1-8b-instruct"}
                label="Model (Name of the model to send in v1/chat/completions payload)"
                name="model"
              />
              <TextField
                defaultValue={tokenizer || "NousResearch/Meta-Llama-3.1-8B-Instruct"}
                label="Tokenizer (HuggingFace tokenizer to use to count tokens)"
                name="tokenizer"
              />
              <TextField
                defaultValue={openaiApiKey || ""}
                label="OpenAI API Key (Required for OpenAI, Anthropic, and other external LLM providers. Skip if using TrueFoundry-hosted models)"
                name="openaiApiKey"
                type="password"
              />
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography>Prompt Configuration</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Box sx={{ display: 'flex', flexDirection: 'column', rowGap: 4 }}>
                    <NumericField
                      defaultValue={maxTokens || 256}
                      label="Max Output Tokens (Maximum number of tokens in the model's response)"
                      name="maxTokens"
                      sx={{ width: '100%' }}
                    />
                    {!promptMode.useSinglePrompt && (
                      <>
                        <NumericField
                          defaultValue={promptMinTokens || 10}
                          label="Prompt Min Tokens (Minimum number of tokens in the input prompt)"
                          name="promptMinTokens"
                          sx={{ width: '100%' }}
                        />
                        <NumericField
                          defaultValue={promptMaxTokens || 1000}
                          label="Prompt Max Tokens (Maximum number of tokens in the input prompt)"
                          name="promptMaxTokens"
                          sx={{ width: '100%' }}
                        />
                      </>
                    )}
                    <FormControlLabel
                      control={
                        <Checkbox
                          checked={promptMode.useRandomPrompts}
                          name="useRandomPrompts"
                          onChange={() => handlePromptModeChange('useRandomPrompts')}
                        />
                      }
                      label="Use Random Prompts (Whether to use randomly generated prompts)"
                    />
                    <FormControlLabel
                      control={
                        <Checkbox
                          checked={promptMode.useSinglePrompt}
                          name="useSinglePrompt"
                          onChange={() => handlePromptModeChange('useSinglePrompt')}
                        />
                      }
                      label="Use Single Prompt, ~1000 tokens of GPT-4o (Whether to use a single prompt for all tests)"
                    />
                    <FormControlLabel
                      control={
                        <Checkbox
                          checked={promptMode.sampleFromDataset}
                          name="sampleFromDataset"
                          onChange={() => handlePromptModeChange('sampleFromDataset')}
                        />
                      }
                      label="Sample from dataset (Whether to sample prompts from a dataset)"
                    />
                    <FormControlLabel
                      control={<Checkbox defaultChecked={ignoreEos || true} name="ignoreEos" />}
                      label="Ignore EOS Token (Generate exactly Max Output Tokens by ignoring end-of-sequence tokens)"
                    />
                  </Box>
                </AccordionDetails>
              </Accordion>
              {!isEmpty(extraOptions) && <CustomParameters extraOptions={extraOptions} />}
            </>
          )}
          {alert && !errorMessage && (
            <Alert severity={alert.level || 'info'}>{alert.message}</Alert>
          )}
          {errorMessage && <Alert severity={'error'}>{errorMessage}</Alert>}
          <Button disabled={isDisabled} size='large' type='submit' variant='contained'>
            {isEditSwarm ? 'Update' : 'Start'}
          </Button>
        </Box>
      </Form>
    </Container>
  );
}

const storeConnector = ({
  swarm: {
    availableShapeClasses,
    availableUserClasses,
    extraOptions,
    hideCommonOptions,
    shapeUseCommonOptions,
    host,
    numUsers,
    userCount,
    overrideHostWarning,
    spawnRate,
    showUserclassPicker,
    tokenizer,
    model,
    maxTokens,
    promptMinTokens,
    promptMaxTokens,
    useRandomPrompts,
    useSinglePrompt,
    ignoreEos,
    openaiApiKey,
  },
}: IRootState) => ({
  availableShapeClasses,
  availableUserClasses,
  extraOptions,
  hideCommonOptions,
  shapeUseCommonOptions,
  host,
  overrideHostWarning,
  showUserclassPicker,
  numUsers,
  userCount,
  spawnRate,
  tokenizer,
  model,
  maxTokens,
  promptMinTokens,
  promptMaxTokens,
  useRandomPrompts,
  useSinglePrompt,
  ignoreEos,
  openaiApiKey,
});

const actionCreator: IDispatchProps = {
  setSwarm: swarmActions.setSwarm,
};

export default connect(storeConnector, actionCreator)(SwarmForm);

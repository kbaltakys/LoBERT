from collections.abc import Mapping, Sized
# from dataclasses import dataclass
import copy
import os
from typing import Dict, List, Optional, Tuple, Union
import warnings

from transformers.tokenization_utils_base import (
    BatchEncoding,
    EncodedInput,
    PaddingStrategy,
    TensorType,
    TruncationStrategy,
    )

from transformers.utils import (
    is_offline_mode,
    logging,
    )


logger = logging.get_logger(__name__)


SPECIAL_TOKENS_MAP_FILE = "special_tokens_map.json"
ADDED_TOKENS_FILE = "added_tokens.json"
TOKENIZER_CONFIG_FILE = "tokenizer_config.json"
CHAT_TEMPLATE_FILE = "chat_template.jinja"


SPECIAL_TOKENS_ATTRIBUTES = [
    "bos_token",
    "eos_token",
    "unk_token",
    "sep_token",
    "pad_token",
    "cls_token",
    "mask_token",
    "additional_special_tokens",
]

# @dataclass
class LoBertTokenizer:
    mask_token: str = "[MASK]"

    mask_token_id: int = 1

    model_input_names: List[str] = ["input_ids", "time_ids", "volume_ids", "token_type_ids", "attention_mask"]
    # TODO We need to define a padding token and its id for padding side to be defined
    pad_token: str = "[PAD]"
    # TODO Make sure the pad token is not overwriting an existing token
    pad_token_id: int = 0

    time_pad_token_id: int = 0
    volume_pad_token_id: int = 0

    padding_side: str = "right"
    truncation_side: str = "right"

    # TODO implement correct length
    def __len__(self):
        return 3200

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:

        if tokens is None:
            return None

        if isinstance(tokens, str):
            if tokens != self.mask_token:
                raise NotImplementedError(
                    "Only the mask token can be converted to an id directly (not the case for token {})".format(tokens)
                )
            return self.mask_token_id

        ids = []
        for token in tokens:
            if token == self.mask_token:
                ids.append(self.mask_token_id)
            else:
                raise NotImplementedError(
                    "Only the mask token can be converted to an id directly (not the case for token {})".format(token)
                )
        return ids

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` or `encode_plus` methods.

        Args:
            token_ids_0 (`List[int]`):
                List of ids of the first sequence.
            token_ids_1 (`List[int]`, *optional*):
                List of ids of the second sequence.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        assert already_has_special_tokens and token_ids_1 is None, (
            "You cannot use ``already_has_special_tokens=False`` with this tokenizer. "
            "Please use a slow (full python) tokenizer to activate this argument. "
            "Or set `return_special_tokens_mask=True` when calling the encoding method "
            "to get the special tokens mask in any tokenizer. "
        )

        all_special_ids = SPECIAL_TOKENS_ATTRIBUTES  # cache the property

        special_tokens_mask = [1 if token in all_special_ids else 0 for token in token_ids_0]

        return special_tokens_mask

    def _get_padding_truncation_strategies(
        self, padding=False, truncation=None, max_length=None, pad_to_multiple_of=None, verbose=True, **kwargs
    ):
        """
        Find the correct padding/truncation strategy with backward compatibility for old arguments (truncation_strategy
        and pad_to_max_length) and behaviors.
        """
        old_truncation_strategy = kwargs.pop("truncation_strategy", "do_not_truncate")
        old_pad_to_max_length = kwargs.pop("pad_to_max_length", False)

        # Backward compatibility for previous behavior, maybe we should deprecate it:
        # If you only set max_length, it activates truncation for max_length
        if max_length is not None and padding is False and truncation is None:
            if verbose:
                if not self.deprecation_warnings.get("Truncation-not-explicitly-activated", False):
                    logger.warning(
                        "Truncation was not explicitly activated but `max_length` is provided a specific value, please"
                        " use `truncation=True` to explicitly truncate examples to max length. Defaulting to"
                        " 'longest_first' truncation strategy. If you encode pairs of sequences (GLUE-style) with the"
                        " tokenizer you can select this strategy more precisely by providing a specific strategy to"
                        " `truncation`."
                    )
                self.deprecation_warnings["Truncation-not-explicitly-activated"] = True
            truncation = "longest_first"

        # Get padding strategy
        if padding is False and old_pad_to_max_length:
            if verbose:
                warnings.warn(
                    "The `pad_to_max_length` argument is deprecated and will be removed in a future version, "
                    "use `padding=True` or `padding='longest'` to pad to the longest sequence in the batch, or "
                    "use `padding='max_length'` to pad to a max length. In this case, you can give a specific "
                    "length with `max_length` (e.g. `max_length=45`) or leave max_length to None to pad to the "
                    "maximal input size of the model (e.g. 512 for Bert).",
                    FutureWarning,
                )
            if max_length is None:
                padding_strategy = PaddingStrategy.LONGEST
            else:
                padding_strategy = PaddingStrategy.MAX_LENGTH
        elif padding is not False:
            if padding is True:
                if verbose:
                    if max_length is not None and (
                        truncation is None or truncation is False or truncation == "do_not_truncate"
                    ):
                        warnings.warn(
                            "`max_length` is ignored when `padding`=`True` and there is no truncation strategy. "
                            "To pad to max length, use `padding='max_length'`."
                        )
                    if old_pad_to_max_length is not False:
                        warnings.warn("Though `pad_to_max_length` = `True`, it is ignored because `padding`=`True`.")
                padding_strategy = PaddingStrategy.LONGEST  # Default to pad to the longest sequence in the batch
            elif not isinstance(padding, PaddingStrategy):
                padding_strategy = PaddingStrategy(padding)
            elif isinstance(padding, PaddingStrategy):
                padding_strategy = padding
        else:
            padding_strategy = PaddingStrategy.DO_NOT_PAD

        # Get truncation strategy
        if truncation is None and old_truncation_strategy != "do_not_truncate":
            if verbose:
                warnings.warn(
                    "The `truncation_strategy` argument is deprecated and will be removed in a future version, use"
                    " `truncation=True` to truncate examples to a max length. You can give a specific length with"
                    " `max_length` (e.g. `max_length=45`) or leave max_length to None to truncate to the maximal input"
                    " size of the model (e.g. 512 for Bert).  If you have pairs of inputs, you can give a specific"
                    " truncation strategy selected among `truncation='only_first'` (will only truncate the first"
                    " sentence in the pairs) `truncation='only_second'` (will only truncate the second sentence in the"
                    " pairs) or `truncation='longest_first'` (will iteratively remove tokens from the longest sentence"
                    " in the pairs).",
                    FutureWarning,
                )
            truncation_strategy = TruncationStrategy(old_truncation_strategy)
        elif truncation is not False and truncation is not None:
            if truncation is True:
                truncation_strategy = (
                    TruncationStrategy.LONGEST_FIRST
                )  # Default to truncate the longest sequences in pairs of inputs
            elif not isinstance(truncation, TruncationStrategy):
                truncation_strategy = TruncationStrategy(truncation)
            elif isinstance(truncation, TruncationStrategy):
                truncation_strategy = truncation
        else:
            truncation_strategy = TruncationStrategy.DO_NOT_TRUNCATE

        # Set max length if needed
        if max_length is None:
            if padding_strategy == PaddingStrategy.MAX_LENGTH:
                if self.model_max_length > LARGE_INTEGER:
                    if verbose:
                        if not self.deprecation_warnings.get("Asking-to-pad-to-max_length", False):
                            logger.warning(
                                "Asking to pad to max_length but no maximum length is provided and the model has no"
                                " predefined maximum length. Default to no padding."
                            )
                        self.deprecation_warnings["Asking-to-pad-to-max_length"] = True
                    padding_strategy = PaddingStrategy.DO_NOT_PAD
                else:
                    max_length = self.model_max_length

            if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE:
                if self.model_max_length > LARGE_INTEGER:
                    if verbose:
                        if not self.deprecation_warnings.get("Asking-to-truncate-to-max_length", False):
                            logger.warning(
                                "Asking to truncate to max_length but no maximum length is provided and the model has"
                                " no predefined maximum length. Default to no truncation."
                            )
                        self.deprecation_warnings["Asking-to-truncate-to-max_length"] = True
                    truncation_strategy = TruncationStrategy.DO_NOT_TRUNCATE
                else:
                    max_length = self.model_max_length

        # Test if we have a padding token
        if padding_strategy != PaddingStrategy.DO_NOT_PAD and (self.pad_token is None or self.pad_token_id < 0):
            raise ValueError(
                "Asking to pad but the tokenizer does not have a padding token. "
                "Please select a token to use as `pad_token` `(tokenizer.pad_token = tokenizer.eos_token e.g.)` "
                "or add a new pad token via `tokenizer.add_special_tokens({'pad_token': '[PAD]'})`."
            )

        # Check that we will truncate to a multiple of pad_to_multiple_of if both are provided
        if (
            truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE
            and padding_strategy != PaddingStrategy.DO_NOT_PAD
            and pad_to_multiple_of is not None
            and max_length is not None
            and (max_length % pad_to_multiple_of != 0)
        ):
            raise ValueError(
                "Truncation and padding are both activated but "
                f"truncation length ({max_length}) is not a multiple of pad_to_multiple_of ({pad_to_multiple_of})."
            )

        return padding_strategy, truncation_strategy, max_length, kwargs


    def _pad(
        self,
        encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
        max_length: Optional[int] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        padding_side: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
    ) -> dict:
        """
        Pad encoded inputs (on left/right and up to predefined length or max length in the batch)

        Args:
            encoded_inputs:
                Dictionary of tokenized inputs (`List[int]`) or batch of tokenized inputs (`List[List[int]]`).
            max_length: maximum length of the returned list and optionally padding length (see below).
                Will truncate by taking into account the special tokens.
            padding_strategy: PaddingStrategy to use for padding.

                - PaddingStrategy.LONGEST Pad to the longest sequence in the batch
                - PaddingStrategy.MAX_LENGTH: Pad to the max length (default)
                - PaddingStrategy.DO_NOT_PAD: Do not pad
                The tokenizer padding sides are defined in `padding_side` argument:

                    - 'left': pads on the left of the sequences
                    - 'right': pads on the right of the sequences
            pad_to_multiple_of: (optional) Integer if set will pad the sequence to a multiple of the provided value.
                This is especially useful to enable the use of Tensor Core on NVIDIA hardware with compute capability
                `>= 7.5` (Volta).
            padding_side:
                The side on which the model should have padding applied. Should be selected between ['right', 'left'].
                Default value is picked from the class attribute of the same name.
            return_attention_mask:
                (optional) Set to False to avoid returning attention mask (default: set to model specifics)
        """
        # Load from model defaults
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        required_input = encoded_inputs[self.model_input_names[0]]

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = len(required_input)

        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(required_input) != max_length

        # Initialize attention mask if not present.
        if return_attention_mask and "attention_mask" not in encoded_inputs:
            encoded_inputs["attention_mask"] = [1] * len(required_input)

        if needs_to_be_padded:
            difference = max_length - len(required_input)
            padding_side = padding_side if padding_side is not None else self.padding_side

            if padding_side == "right":
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = encoded_inputs["attention_mask"] + [0] * difference
                if "token_type_ids" in encoded_inputs:
                    encoded_inputs["token_type_ids"] = (
                        encoded_inputs["token_type_ids"] + [self.pad_token_type_id] * difference
                    )
                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] = encoded_inputs["special_tokens_mask"] + [1] * difference
                encoded_inputs[self.model_input_names[0]] = required_input + [self.pad_token_id] * difference
                encoded_inputs["time_ids"] = encoded_inputs["time_ids"] + [self.time_pad_token_id] * difference
                encoded_inputs["volume_ids"] = encoded_inputs["volume_ids"] + [self.volume_pad_token_id] * difference
            elif padding_side == "left":
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [0] * difference + encoded_inputs["attention_mask"]
                if "token_type_ids" in encoded_inputs:
                    encoded_inputs["token_type_ids"] = [self.pad_token_type_id] * difference + encoded_inputs[
                        "token_type_ids"
                    ]
                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] = [1] * difference + encoded_inputs["special_tokens_mask"]
                encoded_inputs[self.model_input_names[0]] = [self.pad_token_id] * difference + required_input
                encoded_inputs["time_ids"] = [self.time_pad_token_id] * difference + encoded_inputs["time_ids"]
                encoded_inputs["volume_ids"] = [self.volume_pad_token_id] * difference + encoded_inputs["volume_ids"]
            else:
                raise ValueError(f"Invalid padding strategy:{padding_side}")

        return encoded_inputs


    def pad(
        self,
        encoded_inputs: Union[
            BatchEncoding,
            List[BatchEncoding],
            Dict[str, EncodedInput],
            Dict[str, List[EncodedInput]],
            List[Dict[str, EncodedInput]],
        ],
        padding: Union[bool, str, PaddingStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        padding_side: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        verbose: bool = True,
    ) -> BatchEncoding:
        """
        Pad a single encoded input or a batch of encoded inputs up to predefined length or to the max sequence length
        in the batch.

        Padding side (left/right) padding token ids are defined at the tokenizer level (with `self.padding_side`,
        `self.pad_token_id` and `self.pad_token_type_id`).

        Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the
        text followed by a call to the `pad` method to get a padded encoding.

        <Tip>

        If the `encoded_inputs` passed are dictionary of numpy arrays, PyTorch tensors or TensorFlow tensors, the
        result will use the same type unless you provide a different tensor type with `return_tensors`. In the case of
        PyTorch tensors, you will lose the specific device of your tensors however.

        </Tip>

        Args:
            encoded_inputs ([`BatchEncoding`], list of [`BatchEncoding`], `Dict[str, List[int]]`, `Dict[str, List[List[int]]` or `List[Dict[str, List[int]]]`):
                Tokenized inputs. Can represent one input ([`BatchEncoding`] or `Dict[str, List[int]]`) or a batch of
                tokenized inputs (list of [`BatchEncoding`], *Dict[str, List[List[int]]]* or *List[Dict[str,
                List[int]]]*) so you can use this method during preprocessing as well as in a PyTorch Dataloader
                collate function.

                Instead of `List[int]` you can have tensors (numpy arrays, PyTorch tensors or TensorFlow tensors), see
                the note above for the return type.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
                 Select a strategy to pad the returned sequences (according to the model's padding side and padding
                 index) among:

                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see above).
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the sequence to a multiple of the provided value.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                `>= 7.5` (Volta).
            padding_side (`str`, *optional*):
                The side on which the model should have padding applied. Should be selected between ['right', 'left'].
                Default value is picked from the class attribute of the same name.
            return_attention_mask (`bool`, *optional*):
                Whether to return the attention mask. If left to the default, will return the attention mask according
                to the specific tokenizer's default, defined by the `return_outputs` attribute.

                [What are attention masks?](../glossary#attention-mask)
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
            verbose (`bool`, *optional*, defaults to `True`):
                Whether or not to print more information and warnings.
        """
        if self.__class__.__name__.endswith("Fast"):
            if not self.deprecation_warnings.get("Asking-to-pad-a-fast-tokenizer", False):
                logger.warning_advice(
                    f"You're using a {self.__class__.__name__} tokenizer. Please note that with a fast tokenizer,"
                    " using the `__call__` method is faster than using a method to encode the text followed by a call"
                    " to the `pad` method to get a padded encoding."
                )
                self.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

        # If we have a list of dicts, let's convert it in a dict of lists
        # We do this to allow using this method as a collate_fn function in PyTorch Dataloader
        if isinstance(encoded_inputs, (list, tuple)) and isinstance(encoded_inputs[0], Mapping):
            encoded_inputs = {key: [example[key] for example in encoded_inputs] for key in encoded_inputs[0].keys()}

        # The model's main input name, usually `input_ids`, has been passed for padding
        if self.model_input_names[0] not in encoded_inputs:
            raise ValueError(
                "You should supply an encoding or a list of encodings to this method "
                f"that includes {self.model_input_names[0]}, but you provided {list(encoded_inputs.keys())}"
            )

        required_input = encoded_inputs[self.model_input_names[0]]

        if required_input is None or (isinstance(required_input, Sized) and len(required_input) == 0):
            if return_attention_mask:
                encoded_inputs["attention_mask"] = []
            return encoded_inputs

        # If we have PyTorch/TF/NumPy tensors/arrays as inputs, we cast them as python objects
        # and rebuild them afterwards if no return_tensors is specified
        # Note that we lose the specific device the tensor may be on for PyTorch

        first_element = required_input[0]
        if isinstance(first_element, (list, tuple)):
            # first_element might be an empty list/tuple in some edge cases so we grab the first non empty element.
            for item in required_input:
                if len(item) != 0:
                    first_element = item[0]
                    break
        # At this state, if `first_element` is still a list/tuple, it's an empty one so there is nothing to do.
        if not isinstance(first_element, (int, list, tuple)):
            if is_tf_tensor(first_element):
                return_tensors = "tf" if return_tensors is None else return_tensors
            elif is_torch_tensor(first_element):
                return_tensors = "pt" if return_tensors is None else return_tensors
            elif isinstance(first_element, np.ndarray):
                return_tensors = "np" if return_tensors is None else return_tensors
            else:
                raise ValueError(
                    f"type of {first_element} unknown: {type(first_element)}. "
                    "Should be one of a python, numpy, pytorch or tensorflow object."
                )

            for key, value in encoded_inputs.items():
                encoded_inputs[key] = to_py_obj(value)

        # Convert padding_strategy in PaddingStrategy
        padding_strategy, _, max_length, _ = self._get_padding_truncation_strategies(
            padding=padding, max_length=max_length, verbose=verbose
        )

        required_input = encoded_inputs[self.model_input_names[0]]
        if required_input and not isinstance(required_input[0], (list, tuple)):
            encoded_inputs = self._pad(
                encoded_inputs,
                max_length=max_length,
                padding_strategy=padding_strategy,
                pad_to_multiple_of=pad_to_multiple_of,
                padding_side=padding_side,
                return_attention_mask=return_attention_mask,
            )
            return BatchEncoding(encoded_inputs, tensor_type=return_tensors)

        batch_size = len(required_input)
        assert all(
            len(v) == batch_size for v in encoded_inputs.values()
        ), "Some items in the output dictionary have a different batch size than others."

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = max(len(inputs) for inputs in required_input)
            padding_strategy = PaddingStrategy.MAX_LENGTH

        batch_outputs = {}
        for i in range(batch_size):
            inputs = {k: v[i] for k, v in encoded_inputs.items()}
            # input_sizes = {len(v) for k, v in inputs.items()}
            # if max(input_sizes) != 128:
            #     print("Batch encoding inputs have inconsistent lengths. This will lead to a padding of the inputs to the maximum length of the batch. Debug this!")
            outputs = self._pad(
                inputs,
                max_length=max_length,
                padding_strategy=padding_strategy,
                pad_to_multiple_of=pad_to_multiple_of,
                padding_side=padding_side,
                return_attention_mask=return_attention_mask,
            )

            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        return BatchEncoding(batch_outputs, tensor_type=return_tensors)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        *init_inputs,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        trust_remote_code=False,
        **kwargs,
    ):
        r"""
        Instantiate a [`~tokenization_utils_base.PreTrainedTokenizerBase`] (or a derived class) from a predefined
        tokenizer.

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                Can be either:

                - A string, the *model id* of a predefined tokenizer hosted inside a model repo on huggingface.co.
                - A path to a *directory* containing vocabulary files required by the tokenizer, for instance saved
                  using the [`~tokenization_utils_base.PreTrainedTokenizerBase.save_pretrained`] method, e.g.,
                  `./my_model_directory/`.
                - (**Deprecated**, not applicable to all derived classes) A path or url to a single saved vocabulary
                  file (if and only if the tokenizer only requires a single vocabulary file like Bert or XLNet), e.g.,
                  `./my_model_directory/vocab.txt`.
            cache_dir (`str` or `os.PathLike`, *optional*):
                Path to a directory in which a downloaded predefined tokenizer vocabulary files should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download the vocabulary files and override the cached versions if they
                exist.
            resume_download:
                Deprecated and ignored. All downloads are now resumed by default when possible.
                Will be removed in v5 of Transformers.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                when running `huggingface-cli login` (stored in `~/.huggingface`).
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether or not to only rely on local files and not to attempt to download any files.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.
            subfolder (`str`, *optional*):
                In case the relevant files are located inside a subfolder of the model repo on huggingface.co (e.g. for
                facebook/rag-token-base), specify it here.
            inputs (additional positional arguments, *optional*):
                Will be passed along to the Tokenizer `__init__` method.
            trust_remote_code (`bool`, *optional*, defaults to `False`):
                Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
                should only be set to `True` for repositories you trust and in which you have read the code, as it will
                execute code present on the Hub on your local machine.
            kwargs (additional keyword arguments, *optional*):
                Will be passed to the Tokenizer `__init__` method. Can be used to set special tokens like `bos_token`,
                `eos_token`, `unk_token`, `sep_token`, `pad_token`, `cls_token`, `mask_token`,
                `additional_special_tokens`. See parameters in the `__init__` for more details.

        <Tip>

        Passing `token=True` is required when you want to use a private model.

        </Tip>

        Examples:

        ```python
        # We can't instantiate directly the base class *PreTrainedTokenizerBase* so let's show our examples on a derived class: BertTokenizer
        # Download vocabulary from huggingface.co and cache.
        tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")

        # Download vocabulary from huggingface.co (user-uploaded) and cache.
        tokenizer = BertTokenizer.from_pretrained("dbmdz/bert-base-german-cased")

        # If vocabulary files are in a directory (e.g. tokenizer was saved using *save_pretrained('./test/saved_model/')*)
        tokenizer = BertTokenizer.from_pretrained("./test/saved_model/")

        # If the tokenizer uses a single vocabulary file, you can point directly to this file
        tokenizer = BertTokenizer.from_pretrained("./test/saved_model/my_vocab.txt")

        # You can link tokens to special vocabulary when instantiating
        tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased", unk_token="<unk>")
        # You should be sure '<unk>' is in the vocabulary when doing that.
        # Otherwise use tokenizer.add_special_tokens({'unk_token': '<unk>'}) instead)
        assert tokenizer.unk_token == "<unk>"
        ```"""
        resume_download = kwargs.pop("resume_download", None)
        proxies = kwargs.pop("proxies", None)
        use_auth_token = kwargs.pop("use_auth_token", None)
        subfolder = kwargs.pop("subfolder", None)
        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)
        commit_hash = kwargs.pop("_commit_hash", None)
        gguf_file = kwargs.get("gguf_file", None)

        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            token = use_auth_token

        user_agent = {"file_type": "tokenizer", "from_auto_class": from_auto_class, "is_fast": "Fast" in cls.__name__}
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        if is_offline_mode() and not local_files_only:
            logger.info("Offline mode: forcing local_files_only=True")
            local_files_only = True

        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        vocab_files = {}
        init_configuration = {}

        is_local = os.path.isdir(pretrained_model_name_or_path)
        single_file_id = None
        if os.path.isfile(pretrained_model_name_or_path) or is_remote_url(pretrained_model_name_or_path):
            if len(cls.vocab_files_names) > 1 and not gguf_file:
                raise ValueError(
                    f"Calling {cls.__name__}.from_pretrained() with the path to a single file or url is not "
                    "supported for this tokenizer. Use a model identifier or the path to a directory instead."
                )
            warnings.warn(
                f"Calling {cls.__name__}.from_pretrained() with the path to a single file or url is deprecated and "
                "won't be possible anymore in v5. Use a model identifier or the path to a directory instead.",
                FutureWarning,
            )
            file_id = list(cls.vocab_files_names.keys())[0]

            vocab_files[file_id] = pretrained_model_name_or_path
            single_file_id = file_id
        else:
            if gguf_file:
                vocab_files["vocab_file"] = gguf_file
            else:
                # At this point pretrained_model_name_or_path is either a directory or a model identifier name
                additional_files_names = {
                    "added_tokens_file": ADDED_TOKENS_FILE,  # kept only for legacy
                    "special_tokens_map_file": SPECIAL_TOKENS_MAP_FILE,  # kept only for legacy
                    "tokenizer_config_file": TOKENIZER_CONFIG_FILE,
                    # tokenizer_file used to initialize a slow from a fast. Properly copy the `addedTokens` instead of adding in random orders
                    "tokenizer_file": FULL_TOKENIZER_FILE,
                    "chat_template_file": CHAT_TEMPLATE_FILE,
                }
                vocab_files = {**cls.vocab_files_names, **additional_files_names}
                if "tokenizer_file" in vocab_files:
                    # Try to get the tokenizer config to see if there are versioned tokenizer files.
                    fast_tokenizer_file = FULL_TOKENIZER_FILE
                    resolved_config_file = cached_file(
                        pretrained_model_name_or_path,
                        TOKENIZER_CONFIG_FILE,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        resume_download=resume_download,
                        proxies=proxies,
                        token=token,
                        revision=revision,
                        local_files_only=local_files_only,
                        subfolder=subfolder,
                        user_agent=user_agent,
                        _raise_exceptions_for_gated_repo=False,
                        _raise_exceptions_for_missing_entries=False,
                        _raise_exceptions_for_connection_errors=False,
                        _commit_hash=commit_hash,
                    )
                    commit_hash = extract_commit_hash(resolved_config_file, commit_hash)
                    if resolved_config_file is not None:
                        with open(resolved_config_file, encoding="utf-8") as reader:
                            tokenizer_config = json.load(reader)
                            if "fast_tokenizer_files" in tokenizer_config:
                                fast_tokenizer_file = get_fast_tokenizer_file(tokenizer_config["fast_tokenizer_files"])
                    vocab_files["tokenizer_file"] = fast_tokenizer_file

        # Get files from url, cache, or disk depending on the case
        resolved_vocab_files = {}
        unresolved_files = []
        for file_id, file_path in vocab_files.items():
            if file_path is None:
                resolved_vocab_files[file_id] = None
            elif single_file_id == file_id:
                if os.path.isfile(file_path):
                    resolved_vocab_files[file_id] = file_path
                elif is_remote_url(file_path):
                    resolved_vocab_files[file_id] = download_url(file_path, proxies=proxies)
            else:
                resolved_vocab_files[file_id] = cached_file(
                    pretrained_model_name_or_path,
                    file_path,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                    token=token,
                    user_agent=user_agent,
                    revision=revision,
                    subfolder=subfolder,
                    _raise_exceptions_for_gated_repo=False,
                    _raise_exceptions_for_missing_entries=False,
                    _raise_exceptions_for_connection_errors=False,
                    _commit_hash=commit_hash,
                )
                commit_hash = extract_commit_hash(resolved_vocab_files[file_id], commit_hash)

        if len(unresolved_files) > 0:
            logger.info(
                f"Can't load following files from cache: {unresolved_files} and cannot check if these "
                "files are necessary for the tokenizer to operate."
            )

        # If one passes a GGUF file path to `gguf_file` there is no need for this check as the tokenizer will be
        # loaded directly from the GGUF file.
        if all(full_file_name is None for full_file_name in resolved_vocab_files.values()) and not gguf_file:
            raise EnvironmentError(
                f"Can't load tokenizer for '{pretrained_model_name_or_path}'. If you were trying to load it from "
                "'https://huggingface.co/models', make sure you don't have a local directory with the same name. "
                f"Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a directory "
                f"containing all relevant files for a {cls.__name__} tokenizer."
            )

        for file_id, file_path in vocab_files.items():
            if file_id not in resolved_vocab_files:
                continue

            if is_local:
                logger.info(f"loading file {file_path}")
            else:
                logger.info(f"loading file {file_path} from cache at {resolved_vocab_files[file_id]}")

        return cls._from_pretrained(
            resolved_vocab_files,
            pretrained_model_name_or_path,
            init_configuration,
            *init_inputs,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            _commit_hash=commit_hash,
            _is_local=is_local,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        legacy_format: Optional[bool] = None,
        filename_prefix: Optional[str] = None,
        push_to_hub: bool = False,
        **kwargs,
    ) -> Tuple[str]:
        """
        Save the full tokenizer state.


        This method make sure the full tokenizer can then be re-loaded using the
        [`~tokenization_utils_base.PreTrainedTokenizer.from_pretrained`] class method..

        Warning,None This won't save modifications you may have applied to the tokenizer after the instantiation (for
        instance, modifying `tokenizer.do_lower_case` after creation).

        Args:
            save_directory (`str` or `os.PathLike`): The path to a directory where the tokenizer will be saved.
            legacy_format (`bool`, *optional*):
                Only applicable for a fast tokenizer. If unset (default), will save the tokenizer in the unified JSON
                format as well as in legacy format if it exists, i.e. with tokenizer specific vocabulary and a separate
                added_tokens files.

                If `False`, will only save the tokenizer in the unified JSON format. This format is incompatible with
                "slow" tokenizers (not powered by the *tokenizers* library), so the tokenizer will not be able to be
                loaded in the corresponding "slow" tokenizer.

                If `True`, will save the tokenizer in legacy format. If the "slow" tokenizer doesn't exits, a value
                error is raised.
            filename_prefix (`str`, *optional*):
                A prefix to add to the names of the files saved by the tokenizer.
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            kwargs (`Dict[str, Any]`, *optional*):
                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.

        Returns:
            A tuple of `str`: The files saved.
        """
        # TODO Implement saving of the tokenizer!
        return None
        use_auth_token = kwargs.pop("use_auth_token", None)

        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
                FutureWarning,
            )
            if kwargs.get("token", None) is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            kwargs["token"] = use_auth_token

        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        os.makedirs(save_directory, exist_ok=True)

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id = self._create_repo(repo_id, **kwargs)
            files_timestamps = self._get_files_timestamps(save_directory)

        special_tokens_map_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + SPECIAL_TOKENS_MAP_FILE
        )
        tokenizer_config_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + TOKENIZER_CONFIG_FILE
        )
        chat_template_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + CHAT_TEMPLATE_FILE
        )

        tokenizer_config = copy.deepcopy(self.init_kwargs)

        # Let's save the init kwargs
        target_keys = set(self.init_kwargs.keys())
        # Let's save the special tokens map (only the strings)
        target_keys.update(["model_max_length", "clean_up_tokenization_spaces"])

        for k in target_keys:
            if hasattr(self, k):
                tokenizer_config[k] = getattr(self, k)

        # Let's make sure we properly save the special tokens
        tokenizer_config.update(self.special_tokens_map)
        if "extra_special_tokens" not in tokenizer_config:
            tokenizer_config["extra_special_tokens"] = self.extra_special_tokens
            tokenizer_config.update(self.extra_special_tokens)

        saved_raw_chat_template = False
        if self.chat_template is not None:
            if isinstance(self.chat_template, dict):
                # Chat template dicts are saved to the config as lists of dicts with fixed key names.
                # They will be reconstructed as a single dict during loading.
                # We're trying to discourage chat template dicts, and they are always
                # saved in the config, never as single files.
                tokenizer_config["chat_template"] = [{"name": k, "template": v} for k, v in self.chat_template.items()]
            elif kwargs.get("save_raw_chat_template", False):
                with open(chat_template_file, "w", encoding="utf-8") as f:
                    f.write(self.chat_template)
                saved_raw_chat_template = True
                logger.info(f"chat template saved in {chat_template_file}")
                if "chat_template" in tokenizer_config:
                    tokenizer_config.pop("chat_template")  # To ensure it doesn't somehow end up in the config too
            else:
                tokenizer_config["chat_template"] = self.chat_template

        if len(self.init_inputs) > 0:
            tokenizer_config["init_inputs"] = copy.deepcopy(self.init_inputs)
        for file_id in self.vocab_files_names.keys():
            tokenizer_config.pop(file_id, None)

        # no typefields, this way old fast and slow can load it
        tokenizer_config = self.convert_added_tokens(tokenizer_config, add_type_field=True, save=True)

        # Process added tokens seperatly: allows previous versions to ignore it!
        added_tokens = {}
        for key, value in self.added_tokens_decoder.items():
            added_tokens[key] = value.__getstate__()
        tokenizer_config["added_tokens_decoder"] = added_tokens

        # Add tokenizer class to the tokenizer config to be able to reload it with from_pretrained
        tokenizer_class = self.__class__.__name__
        # Remove the Fast at the end unless we have a special `PreTrainedTokenizerFast`
        if tokenizer_class.endswith("Fast") and tokenizer_class != "PreTrainedTokenizerFast":
            tokenizer_class = tokenizer_class[:-4]
        tokenizer_config["tokenizer_class"] = tokenizer_class
        if getattr(self, "_auto_map", None) is not None:
            tokenizer_config["auto_map"] = self._auto_map
        if getattr(self, "_processor_class", None) is not None:
            tokenizer_config["processor_class"] = self._processor_class

        # If we have a custom model, we copy the file defining it in the folder and set the attributes so it can be
        # loaded from the Hub.
        if self._auto_class is not None:
            custom_object_save(self, save_directory, config=tokenizer_config)

        # remove private information
        if "name_or_path" in tokenizer_config:
            tokenizer_config.pop("name_or_path")
            tokenizer_config.pop("special_tokens_map_file", None)
            tokenizer_config.pop("tokenizer_file", None)
        if "device_map" in tokenizer_config:
            tokenizer_config.pop("device_map")

        with open(tokenizer_config_file, "w", encoding="utf-8") as f:
            out_str = json.dumps(tokenizer_config, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
            f.write(out_str)
        logger.info(f"tokenizer config file saved in {tokenizer_config_file}")

        # Sanitize AddedTokens in special_tokens_map

        # kept for forward compatibility, will be removed in transoformers 5. Typefields are not saved for FC, special should not be save either
        write_dict = self.convert_added_tokens(self.special_tokens_map_extended, save=True, add_type_field=False)
        with open(special_tokens_map_file, "w", encoding="utf-8") as f:
            out_str = json.dumps(write_dict, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
            f.write(out_str)
        logger.info(f"Special tokens file saved in {special_tokens_map_file}")

        file_names = (tokenizer_config_file, special_tokens_map_file)
        if saved_raw_chat_template:
            file_names += (chat_template_file,)

        save_files = self._save_pretrained(
            save_directory=save_directory,
            file_names=file_names,
            legacy_format=legacy_format,
            filename_prefix=filename_prefix,
        )

        if push_to_hub:
            self._upload_modified_files(
                save_directory,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=kwargs.get("token"),
            )

        return save_files

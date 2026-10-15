# wetext 0.1.2's normalisation grammars

The FSTs VoxCPM's text normaliser runs (`voxcpm.utils.text_normalize.TextNormalizer`: wetext's `zh` TN with
`remove_erhua`, wetext's `en` TN), copied unchanged from the wheel of
[wetext](https://github.com/pengzhendong/wetext) 0.1.2 (`wetext/fsts/`), the release upstream VoxCPM locks.
They are WeTextProcessing's grammars compiled by wetext's author. Licence: Apache-2.0 (Zhendong Peng).

| File | SHA-256 |
| --- | --- |
| `zh/tn/tagger.fst` | `8261de8e6dbb17f2cc8e51067e237c363b3ffc1341fef4080c2c73b850d46d7f` |
| `zh/tn/verbalizer_remove_erhua.fst` | `f6663ebe403344f9d68c4c6c64d3bf17ba755deda41d3386bc010280b96ee315` |
| `en/tn/tagger.fst` | `bb804f88030153999f154f01950c18dbfd64de2c4cfc5035e21371d4eb4ead8a` |
| `en/tn/verbalizer.fst` | `33ad65460a3e8f7fe313994ff0204a218c0fcb5dec9c09b24aeeaa30ef21d67b` |

They differ from the FSTs CosyVoice's asset holds (ModelScope `pengzhendong/wetext`, which wetext 0.0.4 downloads) and
from WeTextProcessing's own (`mizuoto-indextts`), so each engine keeps the set its upstream ran.

# Text normalisation grammars for IndexTTS

Compiled FSTs of [WeTextProcessing](https://github.com/wenet-e2e/WeTextProcessing) 1.2.0 (Apache-2.0,
© the WeNet community), as IndexTTS normalises text on Linux (`indextts/utils/front.py`):

- `zh_tn_tagger.fst`: the Chinese tagger shipped in the `WeTextProcessing==1.2.0` wheel;
- `zh_tn_verbalizer.fst`: the Chinese verbaliser built by that version with `remove_erhua=False`
  (IndexTTS's `tagger_cache`);
- `en_tn_tagger.fst`, `en_tn_verbalizer.fst`: the English grammars shipped in the wheel.

They are applied with `kaldifst` (`mizuoto_indextts.text.TnNormalizer`), which reproduces pynini's
shortest-path application without pynini, on every platform.

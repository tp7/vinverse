## Vinverse

Vinverse is a simple filter to remove residual combing, based on an [AviSynth script by Didée][1] and originally written by tritical.

This plugin also includes a fast implementation of [Vinverse2 function][2].

### Parameters

* *sstr* - strength of contra sharpening (2.7 by default)
* *amnt* -  change no pixel by more than this (255)
* *uv* - chroma mode, as in MaskTools: 1=trash chroma, 2=pass chroma through, 3=process chroma (3)
* *scl* - scale factor for `VshrpD*VblurD < 0`  (0.25)

  [1]: http://forum.doom9.org/showthread.php?p=841641#post841641
  [2]: http://forum.doom9.org/showthread.php?p=1584186#post1584186

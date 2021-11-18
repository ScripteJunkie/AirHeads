# CHANGES

> Remember --- changes are human readable.

## WIP 2021-11-17 21:30:25

### Removed
* `*.mp4` - see Changed below

### Added
* `README` to mention issues in `assets`

### Changed
* `README` instruction correction
* Uninstall Git-LFS because it is not supported in public forks (see https://github.com/git-lfs/git-lfs/issues/1906)

- - -
## Pull request to upstream - 2021-11-17 18:00:00

Beginning of cleanup to make submodules work.

### Removed
* `launch.json` - weird and wrong file?

### Added
* `CHANGES.md`

### Changed
* Cleaned `gitignore`
* Deleted weird line-ending stuff `gitattributes`
* Moved `assets` to project root
* `*.mp4` to Git-LFS
* Moved `submodule` targets to `lib`
* Moved `old` to `_Attic` as per FP preferences.

- - -

## 2021-11-17 17:00:00

Fork from https://github.com/ScripteJunkie/AirHeads to clean up some stuff.

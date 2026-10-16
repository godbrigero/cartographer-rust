# Architecture report: `common/`, `transform/`, `metrics/`

Source (read-only): `cartographer/cartographer/{common,transform,metrics}/`. Namespaces: `cartographer::common`, `cartographer::common::testing`, `cartographer::transform`, `cartographer::metrics`.
Lua version targeted by upstream: **Lua 5.2** (`package.xml: lua5.2-dev`, `cmake/modules/FindLuaGoogle.cmake`). All "CHECK"/"LOG(FATAL)" = process abort (glog). In Rust, map to `panic!`/`assert!` (or `Result` at API boundaries if you deliberately change semantics; tests use `EXPECT_DEATH` with message regexes, so keep the messages).

---

## 0. File inventory

| Dir | Files | Tests |
|---|---|---|
| common | config.h.cmake, configuration_file_resolver.{h,cc}, fixed_ratio_sampler.{h,cc}, histogram.{h,cc}, lua.h, lua_parameter_dictionary.{h,cc}, math.h, port.h, print_configuration_main.cc (binary), task.{h,cc}, thread_pool.{h,cc}, time.{h,cc}, proto/ceres_solver_options.proto | configuration_files_test.cc, fixed_ratio_sampler_test.cc, lua_parameter_dictionary_test.cc, math_test.cc, task_test.cc, thread_pool_test.cc |
| common/internal | blocking_queue.h, ceres_solver_options.{h,cc}, rate_timer.h | blocking_queue_test.cc, rate_timer_test.cc |
| common/internal/testing | lua_parameter_dictionary_test_helpers.h, thread_pool_for_testing.{h,cc} | (helpers) |
| transform | rigid_transform.{h,cc}, rigid_transform_test_helpers.h, timestamped_transform.{h,cc}, transform.{h,cc}, transform_interpolation_buffer.{h,cc}, proto/transform.proto, proto/timestamped_transform.proto | rigid_transform_test.cc, timestamped_transform_test.cc, transform_test.cc, transform_interpolation_buffer_test.cc |
| metrics | counter.{h,cc}, gauge.{h,cc}, histogram.{h,cc}, family_factory.h, register.{h,cc} | none |

Cross-module deps out of scope: `transform_interpolation_buffer` uses `mapping::proto::Trajectory`; `metrics/register.cc` calls `RegisterMetrics` of mapping/sensor classes; `configuration_files_test` uses `mapping::CreateMapBuilderOptions`/`CreateTrajectoryBuilderOptions`.

---

## 1. common/port.h

| Item | Signature / value | Notes |
|---|---|---|
| int aliases (ns `cartographer`) | `int8..int64`, `uint8..uint64` = `<cstdint>` types | Rust: native `i8..u64`. |
| `RoundToInt` | `int RoundToInt(float)`, `int RoundToInt(double)` | `std::lround` -> round half **away from zero**, truncated to `int`. Rust: `x.round() as i32` (f32/f64::round is half-away-from-zero; `as` saturates whereas C++ is UB on overflow). |
| `RoundToInt64` | `int64 RoundToInt64(float/double)` | same, `as i64`. |
| `FastGzipString` | `void(const std::string& uncompressed, std::string* compressed)` | boost::iostreams `filtering_ostream` with `gzip_compressor(zlib::best_speed)` (level 1) -> `back_inserter(*compressed)`. **Appends** to `*compressed` (does not clear). Footer flushed when stream destroyed. |
| `FastGunzipString` | `void(const std::string& compressed, std::string* decompressed)` | `gzip_decompressor`, appends. |

Users of gzip: `mapping/2d/probability_grid.cc`, `mapping/internal/2d/tsdf_2d.cc`, `mapping/3d/submap_3d.cc`, `io/submap_painter.cc`, `io/proto_stream.cc` (pbstream format!). Rust: `flate2::write::GzEncoder::new(&mut Vec<u8>, Compression::fast())` / `GzDecoder`. Output bytes need not be byte-identical to boost (header fields differ) but must be valid gzip readable by both; for pbstream interop test round-trips against C++-written files. Strings here are binary `Vec<u8>`, not `String`.

## 2. common/config.h.cmake

`constexpr char kConfigurationFilesDirectory[] = "@CARTOGRAPHER_CONFIGURATION_FILES_DIRECTORY@"` (CMake default `${CMAKE_INSTALL_PREFIX}/share/cartographer/configuration_files`); `kSourceDirectory[] = "@PROJECT_SOURCE_DIR@"`. Rust: `env!("CARGO_MANIFEST_DIR")`-relative const or `option_env!` build-time configurable; tests use `kSourceDirectory + "/configuration_files"`.

## 3. common/time.{h,cc}

| Item | Definition |
|---|---|
| `kUtsEpochOffsetFromUnixEpochInSeconds` | `719162 * 24 * 60 * 60` = **62135596800** (int64) - seconds from 0001-01-01 to 1970-01-01. |
| `struct UniversalTimeScaleClock` | `rep = int64`, `period = std::ratio<1, 10000000>` (100 ns ticks), `duration`, `time_point`, `is_steady = true`. No `now()`. |
| `Duration` | `std::chrono::duration<int64, ratio<1,10^7>>` — **signed**. |
| `Time` | `time_point<UniversalTimeScaleClock>`; epoch = 0001-01-01T00:00 UTC. |
| `Duration FromSeconds(double s)` | `duration_cast` from `duration<double>`: computes `(int64)(s * 1e7)` -> **truncation toward zero**. |
| `Duration FromMilliseconds(int64 ms)` | `ms * 10000` ticks. |
| `double ToSeconds(Duration d)` | `(double)ticks / 1e7` (division, not multiply by 1e-7 — matters for bit-exactness). |
| `double ToSeconds(std::chrono::steady_clock::duration d)` | ns (`steady_clock` period on libstdc++/libc++ = 1 ns) `/ 1e9`. |
| `Time FromUniversal(int64 ticks)` / `int64 ToUniversal(Time)` | identity on tick count. |
| `std::ostream& operator<<(ostream&, Time)` | prints `std::to_string(ToUniversal(t))` (integer ticks). |
| `double GetThreadCpuTimeSeconds()` | `clock_gettime(CLOCK_THREAD_CPUTIME_ID)` -> `tv_sec + 1e-9 * tv_nsec`; CHECK on failure (strerror); returns 0 on WIN32. |

Rust: `#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)] struct Time(i64)` and `struct Duration(i64)`; impl `Time - Time = Duration`, `Time ± Duration`, `Duration ± Duration`, `Duration * i64`, `Neg`. **Do not use `std::time::Duration`** (unsigned, ns). `from_seconds(s) = Duration((s * 1e7) as i64)` (Rust `as` truncates toward zero, saturates, NaN->0 — C++ is UB there). `to_seconds = ticks as f64 / 1e7`. CPU time: `libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID)` (available on Linux and macOS >= 10.12). `Display for Time` prints integer.

## 4. common/math.h (header-only templates)

| Function | Signature | Semantics / pitfalls |
|---|---|---|
| `Clamp` | `T Clamp(T value, T min, T max)` | `value>max ? max : value<min ? min : value`. NaN passes through unchanged. (Rust `f64::clamp` panics if min>max and also returns NaN; prefer handwritten for identical semantics.) |
| `Power` | `constexpr T Power(T base, int exponent)` | recursive `base * Power(base, e-1)`, `e==0 -> 1`. **Negative exponent = infinite recursion** (never used). Multiplication order: `b*(b*(b*1))`. |
| `Pow2` | `constexpr T Pow2(T a)` | `Power(a,2)` = `a*(a*1)` = `a*a`. Generic (used with ceres Jets). |
| `DegToRad` | `constexpr double DegToRad(double deg)` | `M_PI * deg / 180.` — evaluate as `(PI*deg)/180` for bit-exactness; `f64::to_radians` computes `deg*(PI/180)` and can differ in last ulp. |
| `RadToDeg` | `constexpr double RadToDeg(double rad)` | `180. * rad / M_PI`. |
| `NormalizeAngleDifference` | `T NormalizeAngleDifference(T d)` | `while (d > pi) d -= 2pi; while (d < -pi) d += 2pi;` Result in **[-pi, pi] inclusive** (pi stays pi, -pi stays -pi, 5pi -> pi, -5pi -> -pi, 2pi -> 0). Port the loops literally; `rem_euclid` differs at boundaries/rounding. Generic over Jets. Infinite loop on inf (NaN returns NaN). |
| `atan2` | `T atan2(const Eigen::Matrix<T,2,1>& v)` | `ceres::atan2(v.y(), v.x())`. |
| `QuaternionProduct` | `void QuaternionProduct(const double* z, const T* w, T* zw)` | Hamilton product with **[w, x, y, z] ordering (ceres convention)**: `zw0 = z0w0 - z1w1 - z2w2 - z3w3; zw1 = z0w1 + z1w0 + z2w3 - z3w2; zw2 = z0w2 - z1w3 + z2w0 + z3w1; zw3 = z0w3 + z1w2 - z2w1 + z3w0`. Note Eigen *storage* order is [x,y,z,w] — do not mix. |

Rust: make generic over a scalar trait (`num_traits::Float` or `nalgebra::RealField`) so autodiff dual numbers (replacement for `ceres::Jet`) work. `M_PI` -> `std::f64::consts::PI`.

## 5. common/fixed_ratio_sampler.{h,cc}

`class FixedRatioSampler { explicit FixedRatioSampler(double ratio); bool Pulse(); std::string DebugString(); }` non-copyable. Fields: `const double ratio_; int64 num_pulses_ = 0, num_samples_ = 0`.
- Ctor: `CHECK_GE(ratio, 0.)`, `LOG_IF(WARNING, ratio == 0.) "FixedRatioSampler is dropping all data."`, `CHECK_LE(ratio, 1.)` (death-test regex `"ratio"` matches the CHECK text).
- `Pulse()`: `++num_pulses_; if ((double)num_samples_ / num_pulses_ < ratio_) { ++num_samples_; return true; } return false;` -> first pulse always true for ratio>0; deterministic, no RNG.
- `DebugString()`: `to_string(num_samples_) + " (" + to_string(100. * num_samples_ / num_pulses_) + "%)"` (to_string double = `%f`; 0 pulses -> NaN text "nan"/"-nan").

## 6. common/histogram.{h,cc} (text histogram, distinct from metrics::Histogram)

`class Histogram { void Add(float value); std::string ToString(int buckets) const; private: std::vector<float> values_; }`
`ToString(buckets)`: `CHECK_GE(buckets,1)`; empty -> `"Count: 0"`. Else min/max/mean (mean = float accumulate from `0.f` / size). `result = StrCat("Count: ", n, "  Min: ", min, "  Max: ", max, "  Mean: ", mean)` (absl float formatting = `%g`-like, 6 significant digits). If `min == max` return. Else `CHECK_LT(min,max)`; for i in 0..buckets: `upper = (i+1==buckets) ? max : max*(i+1)/buckets + min*(buckets-i-1)/buckets` (float arithmetic); count values with `lower <= v && (last ? v <= upper : v < upper)`; append `"\n[%f, %f%c"` with `]` for last else `)`; bar = `(count*20 + n/2) / n` (integer, n is size_t); `"\t"` + 20 chars (`' '` for first `20-bar`, then `'#'`); then `"\tCount: ", count, " (", count*1e2f/n, "%)", "\tTotal: ", total, " (", total*1e2f/n, "%)"`; `lower = upper`. Rust: output is debug-only; exact formatting matters only if tests compare strings (none do). Implement a `%g`-style formatter if exactness wanted.

## 7. common/lua.h

Just `#include <lua.hpp>`.

## 8. common/lua_parameter_dictionary.{h,cc}  (the config system — port carefully)

### API
```cpp
class FileResolver {               // interface
  virtual std::string GetFullPathOrDie(const std::string& basename) = 0;
  virtual std::string GetFileContentOrDie(const std::string& basename) = 0;
};
class LuaParameterDictionary {     // non-copyable
  LuaParameterDictionary(const std::string& code, std::unique_ptr<FileResolver>);   // ReferenceCount::YES
  static std::unique_ptr<LuaParameterDictionary> NonReferenceCounted(const std::string& code, std::unique_ptr<FileResolver>);
  ~LuaParameterDictionary();
  std::vector<std::string> GetKeys() const;
  bool HasKey(const std::string& key) const;
  std::string GetString(const std::string& key);
  double GetDouble(const std::string& key);
  int GetInt(const std::string& key);
  bool GetBool(const std::string& key);
  std::unique_ptr<LuaParameterDictionary> GetDictionary(const std::string& key);
  int GetNonNegativeInt(const std::string& key);
  std::string ToString() const;
  std::vector<double> GetArrayValuesAsDoubles();
  std::vector<std::string> GetArrayValuesAsStrings();
  std::vector<std::unique_ptr<LuaParameterDictionary>> GetArrayValuesAsDictionaries();
 private:
  enum class ReferenceCount { YES, NO };  enum class Quoted { YES, NO };
  lua_State* L_; int index_into_reference_table_;
  const std::shared_ptr<FileResolver> file_resolver_;   // shared with all sub-dicts
  const ReferenceCount reference_count_;
  std::map<std::string,int> reference_counts_;
  std::vector<std::string> included_files_;           // only meaningful on root
};
```

### Construction (root)
1. `L_ = luaL_newstate()`, `index_into_reference_table_ = -1`, CHECK non-null.
2. Registry `"this"` = `lua_pushlightuserdata(this)` (`lua_settable(L, LUA_REGISTRYINDEX)`). Only the root sets it; `include`/`read` always resolve the **root** dict.
3. `luaL_openlibs` (full stdlib incl. `math`, `string`, `os`, `io`).
4. Registers globals: `choose`, `include`, `read` (`lua_register`).
5. `luaL_loadstring(code)` + `lua_pcall(L,0,1,0)`; error -> `CHECK_EQ(status,0) << lua_tostring(L,-1)`.
6. CHECK top of stack is a table ("Topmost item on Lua stack is not a table!"). The table **stays on the stack**; values are read lazily (type known only at Get* time).

### Sub-dictionary (private ctor, from `PopDictionary`)
`L_ = lua_newthread(parent_L)` (pushes thread on parent stack), `luaL_ref(parent_L, LUA_REGISTRYINDEX)` pops it into registry (prevents GC; index > 0), then CHECK parent top is table and `lua_xmove(parent_L, L_, 1)` moves the table onto the coroutine's stack — parent stack is back to its own table on top. Shares `file_resolver_` (shared_ptr), inherits `reference_count` given by caller.

### Destructor
If `reference_count_ == YES` -> `CheckAllKeysWereUsedExactlyOnceAndReset()`. If `index_into_reference_table_ > 0` -> `luaL_unref(L_, LUA_REGISTRYINDEX, idx)` else `lua_close(L_)`.
**Ownership hazard:** sub-dicts hold a coroutine of the root's `lua_State`; they must be destroyed before the root (root closes the state). `print_configuration_main` keeps a vector "stack" for this reason. In Rust, encode this with a shared `Rc<Lua>` (mlua `Lua` is internally refcounted, so sub-dicts holding a `mlua::Table` keep the state alive — this removes the hazard).

### Key/value semantics
| Method | Behavior |
|---|---|
| `GetKeys()` | `lua_next` iteration over the table; pops value; keeps key if `!lua_isnumber(key)` -> returns **string keys only**, in **Lua hash order (unspecified)**. Pitfall: `lua_isnumber` is true for numeric strings too, so a string key `"1"` is excluded. Uses `lua_tostring` on key (safe since not a number). |
| `HasKey(key)` | `lua_rawget` with string key; true iff non-nil. |
| `CheckHasKey` | `CHECK(HasKey(key)) << "Key '" << key << "' not in dictionary:\n" << ToString()` (death regex `"Key.* not in dictionary."`). |
| `CheckHasKeyAndReference` | CheckHasKey + `reference_counts_[key]++`. Called by every `Get*(key)`, regardless of ReferenceCount mode. |
| `GetString` | rawget; `CHECK(lua_isstring)` "Top of stack is not a string value." — **numbers accepted** and converted by `lua_tostring` (Lua 5.2: `%.14g`, so `10` -> `"10"`, test `GetDictionary` relies on `GetString("fasel")=="10"`). |
| `GetDouble` | `CHECK(lua_isnumber)` "Top of stack is not a number value." (numeric strings also accepted, converted); `lua_tonumber`. |
| `GetInt` | same check; `lua_tointeger` then cast to `int`. Lua 5.2 on x86_64 = C cast (truncation toward zero) of the double. Lua 5.3+ would return 0 for non-integral floats — behavior difference if you use mlua with lua53/54. |
| `GetBool` | `CHECK(lua_isboolean)` "Top of stack is not a boolean value." strict (no truthiness). |
| `GetDictionary` | references key, rawget, `PopDictionary(reference_count_)` -> child with same mode. CHECK value is table. |
| `GetNonNegativeInt` | `GetInt` (counts as one reference) then `CHECK_GE(v,0) << v << " is negative."`. |
| `GetArrayValues*` | iterate `idx = 1,2,...` with `lua_rawget` until first nil (stops at first hole). **No reference counting** (integer keys are invisible to `GetKeys`, so the unused-key check ignores them). Doubles: `PopDouble`; Strings: `PopString(NO)`; Dictionaries: `PopDictionary(reference_count_)` (children are themselves ref-checked on destruction). |
| `CheckAllKeysWereUsedExactlyOnceAndReset` | for every key in `GetKeys()`: `CHECK_EQ(1, reference_counts_.count(key))` and `CHECK_EQ(1, reference_counts_.at(key))` "Key 'k' was used the wrong number of times." then `clear()`. => every string key must be read **exactly once**: unused key aborts, double read aborts. Keys referenced that don't exist already aborted earlier. Mode NO skips this entirely (and children inherit NO). |

### ToString / DoToString(indent)
- Output starts `"{"`. First, array part `i = 1..` until nil: `"\n" + indent + "  " + value + ","`. Then string keys from `GetKeys()` **sorted** (std::sort, bytewise): `"\n" + indent + "  " + key + " = " + value + ","`. Then `"\n" + indent + "}"`. If no entries at all -> `"{}"`.
- Value formatting by `lua_type`: boolean -> `true`/`false`; string -> Lua `string.format("%q", s)` (via `QuoteStringOnStack`: calls global `string.format`); number -> `std::to_string(double)` i.e. `%f` 6 decimals (`123` -> `123.000000`, `3.1` -> `3.100000`), `±inf` -> `math.huge` / `-math.huge` (NaN -> "nan"/"-nan", not re-parseable); table -> recursive `DoToString(indent + "  ")` on a **NonReferenceCounted** child; anything else -> `LOG(FATAL) "Unhandled type <typename>"`.
- Does not touch `reference_counts_` (uses Pop* directly). Output is valid Lua when prefixed by `return ` (round-trip test).
- `%q` in 5.2 escapes `"`, `\`, newline -> `\` + actual newline, `\r` -> `\r`, `\0` -> `\0`, other control chars -> `\ddd`. If you don't call Lua's `string.format`, reimplement exactly.

### Lua builtins registered
| Name | Behavior |
|---|---|
| `choose(cond, a, b)` | `CHECK_EQ(lua_gettop,3) "choose() takes (condition, a, b)."`; `CHECK(lua_isboolean(1)) "condition is not a boolean value."`; returns a if true else b (both already evaluated). |
| `include(basename)` | `CHECK_EQ(top,1)`, `CHECK(lua_isstring) "include takes a filename."`. Root dict from registry. `filename = file_resolver_->GetFullPathOrDie(basename)`; if `filename` already in `included_files_` -> `LOG(FATAL) "Tried to include <f> twice. Already included files in order of inclusion: "` + each file + `"\n"`. Push to `included_files_`. Pop arg, `CHECK_EQ(top,0)`. `content = GetFileContentOrDie(basename)` (resolved a 2nd time), `luaL_loadbuffer(content, size, chunkname=filename)`, `lua_pcall(0, LUA_MULTRET)`; returns all chunk results (`lua_gettop`). Included code runs in the **same global environment** — this is how `include "map_builder.lua"` defines global `MAP_BUILDER`. Double-include is fatal (no include-guard semantics). |
| `read(basename)` | CHECKs as above ("read takes a filename."); returns `GetFileContentOrDie(basename)` as a Lua string via `lua_pushstring` (truncates at first NUL byte). |

Lua C API calls used: `luaL_newstate, luaL_openlibs, lua_register, luaL_loadstring, luaL_loadbuffer, lua_pcall, lua_call, lua_newthread, luaL_ref, luaL_unref, lua_xmove, lua_close, lua_pushnil, lua_next, lua_pop, lua_rawget, lua_gettable, lua_settable, lua_pushstring, lua_pushinteger, lua_pushvalue, lua_pushlightuserdata, lua_pushglobaltable, lua_getfield, lua_replace, lua_gettop, lua_type, lua_typename, lua_isnil, lua_isnumber, lua_isstring, lua_isboolean, lua_istable, lua_isthread, lua_tonumber, lua_tointeger, lua_toboolean, lua_tostring, lua_touserdata`, `LUA_REGISTRYINDEX`, `LUA_MULTRET`.

### Rust porting notes (LuaParameterDictionary)
- Use `mlua` with feature `lua52` + `vendored` for behavior parity (numeric->string coercion `%.14g`, all numbers are doubles, `lua_tointeger` truncation). With `lua54`, integer/float subtypes change `tostring` (`10.0` -> `"10.0"`) and `tointeger` of `1.5` fails.
- Struct: `{ lua: mlua::Lua (or Rc), table: mlua::Table, file_resolver: Rc<dyn FileResolver> (Arc if Send needed), reference_count: bool, reference_counts: RefCell<BTreeMap<String,i32>>, included_files: Rc<RefCell<Vec<String>>> }`. Getters take `&self` with interior mutability, or `&mut self` to mirror C++ (non-const getters). Implement unused-key check in `Drop` (beware: panicking in `Drop` during unwinding aborts; guard with `!std::thread::panicking()`).
- `include`/`read` closures: register via `lua.create_function` capturing the resolver + included-files list (replaces the registry `"this"` light-userdata trick).
- Use raw access (`Table::raw_get`) to match `lua_rawget` (no metamethods).
- Provide `dictionary_from_string`-style helpers for tests (C++ `MakeDictionary` in test helpers uses `DummyFileResolver` whose methods `LOG(FATAL) "Not implemented"`).
- Not thread-safe (single lua_State); `mlua::Lua` is `!Send` unless feature `send` is enabled. Options are parsed once into protos/structs, so single-threaded use is fine.

## 9. common/configuration_file_resolver.{h,cc}

`class ConfigurationFileResolver : public FileResolver { explicit ConfigurationFileResolver(const std::vector<std::string>& dirs); GetFullPathOrDie; GetFileContentOrDie; }`
- Ctor copies dirs and **appends `kConfigurationFilesDirectory` last**.
- `GetFullPathOrDie(basename)`: for each dir in order, `filename = dir + "/" + basename`; first where `std::ifstream(filename).good()` -> `LOG(INFO) "Found '<f>' for '<b>'."`, return. Else `LOG(FATAL) "File '<b>' was not found."`.
- `GetFileContentOrDie(basename)`: `CHECK(!basename.empty()) "File basename cannot be empty."`, resolve, read whole file (bytes).
Rust: `Vec<PathBuf>`, `Path::join` (note: C++ uses string concat; `join` with an absolute basename replaces the dir — keep string concat for parity), `File::open(..).is_ok()`, `fs::read_to_string`, `log::info!`.

## 10. common/print_configuration_main.cc (binary)

gflags: `--configuration_directories` (comma-separated, `SkipEmpty`), `--configuration_basename`, `--subdictionary` (dot-separated path). `LoadLuaDictionary(dirs, basename)` = read file content via resolver, build **NonReferenceCounted** dict. If no subdictionary: print `"return " + ToString()`. Else `PrintSubdictionaryById`: split on `.`, CHECK non-empty "Failed to parse 'subdictionary_id'.", chain `GetDictionary` keeping all intermediates alive in a vector, print `"<id> = " + ToString()`. Missing required flags -> usage, `EXIT_FAILURE`. Rust: `clap` binary.

## 11. common/proto/ceres_solver_options.proto + internal/ceres_solver_options.{h,cc}

`message CeresSolverOptions { bool use_nonmonotonic_steps = 1; int32 max_num_iterations = 2; int32 num_threads = 3; }` (proto3, package `cartographer.common.proto`).
- `proto::CeresSolverOptions CreateCeresSolverOptionsProto(LuaParameterDictionary*)`: `GetBool("use_nonmonotonic_steps")`, `GetNonNegativeInt("max_num_iterations")`, `GetNonNegativeInt("num_threads")`; `CHECK_GT(max_num_iterations, 0)`, `CHECK_GT(num_threads, 0)`.
- `ceres::Solver::Options CreateCeresSolverOptions(const proto&)`: copies the three fields into a default `ceres::Solver::Options`.
Rust: `prost`-generated struct; second function maps to whatever optimizer replaces Ceres (e.g. own LM solver / `argmin` / `tiny-solver`) — keep defaults of ceres in mind (trust region LM, etc.) when that module is ported.

## 12. common/task.{h,cc} + common/thread_pool.{h,cc}  (task dependency mechanism)

### API
```cpp
class Task {                                     // friend ThreadPoolInterface
  using WorkItem = std::function<void()>;
  enum State { NEW, DISPATCHED, DEPENDENCIES_COMPLETED, RUNNING, COMPLETED };
  State GetState();                              // locks
  void SetWorkItem(const WorkItem&);             // CHECK state==NEW
  void AddDependency(std::weak_ptr<Task> dep);   // CHECK state==NEW; expired/null dep = already completed
 private:
  void AddDependentTask(Task* dependent);        // any state
  void Execute();                                // DEPENDENCIES_COMPLETED -> RUNNING -> COMPLETED
  void SetThreadPool(ThreadPoolInterface*);      // NEW -> DISPATCHED | DEPENDENCIES_COMPLETED
  void OnDependenyCompleted();                   // [sic] NEW|DISPATCHED
  WorkItem work_item_; ThreadPoolInterface* thread_pool_to_notify_ = nullptr;
  State state_ = NEW; unsigned uncompleted_dependencies_ = 0;
  std::set<Task*> dependent_tasks_; absl::Mutex mutex_;
};
class ThreadPoolInterface {
  virtual std::weak_ptr<Task> Schedule(std::unique_ptr<Task>) = 0;
 protected: void Execute(Task*); void SetThreadPool(Task*);      // trampolines to private Task methods
 private: friend class Task; virtual void NotifyDependenciesCompleted(Task*) = 0;
};
class ThreadPool : public ThreadPoolInterface {  // non-copyable
  explicit ThreadPool(int num_threads); ~ThreadPool();
  std::weak_ptr<Task> Schedule(std::unique_ptr<Task>) override;
 private: void DoWork(); void NotifyDependenciesCompleted(Task*) override;
  absl::Mutex mutex_; bool running_ = true; std::vector<std::thread> pool_;
  std::deque<std::shared_ptr<Task>> task_queue_;
  absl::flat_hash_map<Task*, std::shared_ptr<Task>> tasks_not_ready_;
};
```

### State machine / algorithm
- `AddDependency(weak dep)`: under own lock, CHECK NEW; if `dep.lock()` succeeds, `++uncompleted_dependencies_`. After releasing own lock: `dep->AddDependentTask(this)`.
- `AddDependentTask(dependent)` (under dep's lock): if dep `COMPLETED` -> immediately `dependent->OnDependenyCompleted()` (decrements back). Else insert raw pointer into `dependent_tasks_`; `CHECK(inserted) "Given dependency is already a dependency."` (adding same dependency twice aborts).
- `Schedule(unique_ptr<Task>)` (ThreadPool): under pool lock, move into `tasks_not_ready_` keyed by raw pointer (`CHECK(inserted) "Schedule called twice"`), keep a `shared_ptr` copy; release lock; `SetThreadPool(task)`; return `weak_ptr`.
- `SetThreadPool(pool)` (task lock): CHECK NEW; state = DISPATCHED; store pool; if `uncompleted == 0` -> state = DEPENDENCIES_COMPLETED and `pool->NotifyDependenciesCompleted(this)` (still holding task lock).
- `OnDependenyCompleted()` (task lock): CHECK NEW or DISPATCHED; `--uncompleted`; if 0 and DISPATCHED -> DEPENDENCIES_COMPLETED + notify pool. (If still NEW, it'll be picked up in `SetThreadPool`.)
- `NotifyDependenciesCompleted(task)` (pool lock): find in `tasks_not_ready_` (CHECK found), move shared_ptr to back of `task_queue_`, erase.
- `DoWork()` per thread: Linux only `CHECK_NE(nice(10), -1)` (lower priority). Loop: lock, `Await(!task_queue_.empty() || !running_)`; pop front if any, else if `!running_` return. Unlock; `CHECK(task)`; `CHECK_EQ(state, DEPENDENCIES_COMPLETED)`; `Execute(task)`. The local `shared_ptr` dies after execution -> **the returned weak_ptr expires once the task has run** (that's the contract: expired weak == certainly completed).
- `Execute()`: lock; CHECK DEPENDENCIES_COMPLETED; RUNNING; unlock; run `work_item_` if set (outside lock); lock; COMPLETED; for each dependent: `dependent->OnDependenyCompleted()` (holding own lock).
- `~Task()`: if state not NEW and not COMPLETED -> `LOG(WARNING) "Delete Task between dispatch and completion."`.
- `ThreadPool(int n)`: `CHECK_GT(n,0) "ThreadPool requires a positive num_threads!"`; spawns n threads. `~ThreadPool()`: CHECK running_, set false, join all. Workers drain the ready queue before exiting (they only return when queue empty and !running). Tasks still in `tasks_not_ready_` are just dropped (doc says queue must be empty before destruction).
- FIFO order of ready tasks; no priorities; no work stealing.

### Lock ordering / threading
Order: dependency-task lock -> dependent-task lock -> pool lock (from `Execute`/`AddDependentTask` -> `OnDependenyCompleted` -> `NotifyDependenciesCompleted`); `SetThreadPool`: task lock -> pool lock. Pool never calls into a task while holding the pool lock (Schedule releases first; DoWork executes outside). Keep this order in Rust to avoid deadlock. `absl::Mutex::Await(Condition)` re-evaluates predicates automatically on unlock — with `std::sync::Condvar` you must `notify_one/all` after each push and on shutdown (`notify_all`).

### Ownership
Before Schedule: caller owns `unique_ptr<Task>`; after: pool owns `shared_ptr`; caller only has `weak_ptr`. Dependents are stored as **raw `Task*`** in their dependencies; safe because a dependent cannot be destroyed before it runs if it's scheduled (pool holds it), but if a caller adds dependencies to a task and then drops it without scheduling, the dependency holds a dangling pointer (UB in C++). Work items capture by reference (`[&receiver]`, `this`) — lifetime is the caller's responsibility.

### Rust mapping
- `Task` as `Arc<Task>` from creation (needed because dependents register themselves before being scheduled; stable address). Inner `Mutex<TaskInner { work_item: Option<Box<dyn FnOnce() + Send>>, pool: Option<Weak<dyn ThreadPoolInterface>> or Arc, state, uncompleted: u32, dependents: Vec<Arc<Task>> (or Weak) }>`. Use `Weak<Task>` for the handle returned by `schedule`. Storing `Arc` of dependents in the dependency is fine (no cycles: dependency -> dependent only) and removes the dangling-pointer hazard.
- `WorkItem` is `std::function<void()>` (copyable, callable once in practice) -> `Box<dyn FnOnce() + Send + 'static>`; captures that borrow (`&receiver`) need `Arc` or `std::thread::scope`.
- `tasks_not_ready_` keyed by `Arc::as_ptr` (`HashMap<*const Task, Arc<Task>>`) or by an id.
- `ThreadPoolInterface` -> trait with `schedule(&self, Arc<Task>) -> Weak<Task>` and `notify_dependencies_completed(&self, &Arc<Task>)`; `execute`/`set_thread_pool` become `pub(crate)` free fns.
- `nice(10)`: `libc::nice(10)` under `#[cfg(target_os = "linux")]`.
- Alternatives (rayon) don't provide this dependency semantics; implement directly (~150 LOC).

## 13. common/internal/blocking_queue.h

`template <typename T> class BlockingQueue` ('T' movable; in practice `unique_ptr`/`shared_ptr`). `static constexpr size_t kInfiniteQueueSize = 0;` ctors `BlockingQueue()` (infinite), `explicit BlockingQueue(size_t queue_size)`; non-copyable. Members: `absl::Mutex mutex_; const size_t queue_size_; std::deque<T> deque_`.
| Method | Behavior |
|---|---|
| `void Push(T)` | block until not full (`queue_size_==0 || size < queue_size_`), push_back. |
| `bool PushWithTimeout(T, Duration)` | wait with timeout (`absl::FromChrono`); false on timeout (value dropped). |
| `T Pop()` | block until non-empty, pop_front. |
| `T PopWithTimeout(Duration)` | returns `nullptr` on timeout (so T must be pointer-like). |
| `template<R> R* PeekWithTimeout(Duration)` | `deque_.front().get()` or nullptr on timeout. |
| `template<R> const R* Peek()` | front `.get()` or nullptr if empty; non-blocking. |
| `size_t Size()` | locked size. |
| `void WaitUntilEmpty()` | block until empty. |
Rust: `Mutex<VecDeque<T>>` + `Condvar not_empty`, `Condvar not_full` (or one with `notify_all`); `pop_timeout -> Option<T>`, `peek(&self, f: impl FnOnce(&T) -> R) -> Option<R>` (can't return a reference out of a lock; or return a guard). `wait_timeout_while` for timeouts. Alternatively `crossbeam_channel::bounded/unbounded` covers Push/Pop/timeouts but not Peek/WaitUntilEmpty/Size semantics exactly — prefer custom. Note `Duration` here is common::Duration (100ns ticks) -> convert to `std::time::Duration`.

## 14. common/internal/rate_timer.h

`template <typename ClockType = std::chrono::steady_clock> class RateTimer` (non-copyable). `explicit RateTimer(common::Duration window_duration)`. `struct Event { common::Time time; ClockType::time_point wall_time; }; std::deque<Event> events_;`
- `void Pulse(Time t)`: push `{t, ClockType::now()}`; then `while (size > 2 && back.wall - front.wall > window) pop_front` (keeps ≥2 events). Comparison `steady_clock::duration > common::Duration` via chrono common type.
- `double ComputeRate()`: empty -> 0; else `(size-1) / ToSeconds(back.time - front.time)` (data-time rate; 1 event -> 0/0 = NaN).
- `double ComputeWallTimeRateRatio()`: empty -> 0; `ToSeconds(data span) / ToSeconds(wall span)`.
- `std::string DebugString()`: size<2 -> `"unknown"`; else fixed precision 2: `"<rate> Hz <mean> s +/- <sigma> s (pulsed at <ratio*100>% real time)"` where deltas mean/sigma printed in `std::scientific` precision 2 (sigma = sample stddev, divides by n-1 -> NaN for 1 delta). `ComputeDeltasInSeconds` CHECK_GT(size,1).
Rust: generic over a `Clock` trait (`fn now() -> Instant`-like) to allow simulated clock in tests; default `std::time::Instant`. `{:.2}`/`{:.2e}` formatting (Rust `e` notation differs from C `1.00e-01`: Rust prints `1.00e-1`; only debug output).

## 15. common/internal/testing/*

- `lua_parameter_dictionary_test_helpers.h`: `class DummyFileResolver : FileResolver` (both methods `LOG(FATAL) "Not implemented"`); `std::unique_ptr<LuaParameterDictionary> MakeDictionary(const std::string& code)` (reference counted, dummy resolver). Used widely by other modules' tests.
- `thread_pool_for_testing.{h,cc}`: `class testing::ThreadPoolForTesting : ThreadPoolInterface` — single worker thread; `Schedule` (same as ThreadPool but sets `idle_=false`, CHECK running_, msg "ScheduleWhenReady called twice"), `void WaitUntilIdle()` (loop: AwaitWithTimeout(idle_, 0.1 s)), `NotifyDependenciesCompleted` (CHECK running_). `DoWork`: AwaitWithTimeout(queue non-empty or !running, 0.1 s); pop front if any; return if !running (note: checks running **after** popping, so a popped task may be dropped at shutdown); sets `idle_ = true` when `tasks_not_ready_` empty, queue empty and no task popped; executes outside lock. Destructor: CHECK running, set false, `CHECK_EQ(task_queue_.size(),0)`, `CHECK_EQ(tasks_not_ready_.size(),0)`, join. Uses `std::map<Task*, shared_ptr<Task>>`. Used by mapping tests (pose graph) for deterministic stepping.

## 16. transform/proto/*.proto (package `cartographer.transform.proto`, proto3)

| Message | Fields |
|---|---|
| Vector2d / Vector2f | x=1, y=2 (double/float) |
| Vector3d / Vector3f | x=1, y=2, z=3 |
| Vector4f | x=1, y=2, z=3, **t=4** (maps to Eigen `w()`) |
| Quaterniond / Quaternionf | **x=1, y=2, z=3, w=4** |
| Rigid2d / Rigid2f | translation(Vector2*)=1, rotation(double/float angle rad)=2 |
| Rigid3d / Rigid3f | translation(Vector3*)=1, rotation(Quaternion*)=2 |
| TimestampedTransform | int64 time=1 (UTS ticks), Rigid3d transform=2 |
Frame convention: right-handed, x forward, y left, z up. Rust: `prost` + `prost-build`; proto3 default for a missing Quaterniond is all zeros (w=0!) -> `ToRigid3` of a default proto yields a zero quaternion (not identity). Preserve.

## 17. transform/rigid_transform.{h,cc}

### `template <typename FloatType> class Rigid2`
Types: `Vector = Eigen::Matrix<F,2,1>`, `Rotation2D = Eigen::Rotation2D<F>` (stores a **raw angle scalar**, not normalized).
| Member | Semantics |
|---|---|
| `Rigid2()` | t=0, angle 0 |
| `Rigid2(const Vector&, const Rotation2D&)`, `Rigid2(const Vector&, double rotation)` | angle stored as given (double narrowed to F) |
| `static Rotation(double)`, `Rotation(const Rotation2D&)`, `Translation(const Vector&)`, `Identity()` | |
| `template<O> Rigid2<O> cast() const` | casts translation & angle |
| `const Vector& translation()`, `Rotation2D rotation()` | |
| `double normalized_angle()` | `NormalizeAngleDifference(rotation().angle())` |
| `Rigid2 inverse()` | `rot' = Rotation2D(-angle)`; `t' = -(rot' * t)` |
| `std::string DebugString()` | `absl::Substitute("{ t: [$0, $1], r: [$2] }", x, y, angle)` |
| `operator*(Rigid2, Rigid2)` | `{ lhs.rot * rhs.t + lhs.t, Rotation2D(lhs.angle + rhs.angle) }` — angles **add, no wrapping** |
| `operator*(Rigid2, Vector)` | `rot * p + t`, rot matrix `[[cos,-sin],[sin,cos]]` |
| `operator<<` | DebugString |
Aliases `Rigid2d`, `Rigid2f`.

### `template <typename FloatType> class Rigid3`
Types: `Vector = Matrix<F,3,1>`, `Quaternion = Eigen::Quaternion<F>`, `AngleAxis = Eigen::AngleAxis<F>`.
| Member | Semantics |
|---|---|
| `Rigid3()` | t=0, q=identity |
| `Rigid3(const Vector&, const Quaternion&)` | **no normalization** |
| `Rigid3(const Vector&, const AngleAxis&)` | q = `(cos(a/2), sin(a/2)*axis)` (axis assumed unit) |
| `static Rotation(const AngleAxis&)`, `Rotation(const Quaternion&)`, `Translation(const Vector&)`, `Identity()` | |
| `static FromArrays(const std::array<F,4>& rotation, const std::array<F,3>& translation)` | rotation array order **[w, x, y, z]** (Eigen ctor order) |
| `cast<O>()` | |
| `translation()`, `rotation()` (const refs) | |
| `Rigid3 inverse()` | `q' = q.conjugate()` (assumes unit), `t' = -(q' * t)` |
| `DebugString()` | `"{ t: [$0, $1, $2], q: [$3, $4, $5, $6] }"` with q printed as **w, x, y, z** |
| `bool IsValid()` | no NaN in translation x,y,z AND `|1 - q.norm()| < 1e-3` (does not check quaternion NaN explicitly; NaN norm -> comparison false -> invalid) |
| `operator*(Rigid3, Rigid3)` | `{ lhs.q * rhs.t + lhs.t, (lhs.q * rhs.q).normalized() }` — **renormalizes** (Rigid2 has no analogue) |
| `operator*(Rigid3, Vector)` | `q * p + t` |
Aliases `Rigid3d`, `Rigid3f`.

Eigen behaviors to replicate exactly:
- `Quaternion * Vector3` (Eigen `_transformVector`): `uv = 2*(q.vec × v); return v + q.w*uv + q.vec × uv;` — valid only for unit q; for non-unit q the result is NOT `q v q*`. nalgebra `UnitQuaternion * Vector3` uses an equivalent formula; but if you keep a raw `Quaternion`, implement this formula yourself.
- `Quaternion * Quaternion`: Hamilton product, not normalized.
- `normalized()`: `coeffs / norm` (returns unchanged if norm == 0 in Eigen ≥3.3).
- `conjugate()`: negate vec.
- Eigen quaternion ctor `(w, x, y, z)`; storage/`coeffs()` order `[x, y, z, w]`. nalgebra: `Quaternion::new(w, i, j, k)`, `coords = [i, j, k, w]` — same conventions.

### Free functions (rigid_transform.cc)
- `Eigen::Quaterniond RollPitchYaw(double roll, double pitch, double yaw)` = `AngleAxis(yaw, Z) * AngleAxis(pitch, Y) * AngleAxis(roll, X)` (URDF / extrinsic XYZ = intrinsic ZYX). nalgebra `UnitQuaternion::from_euler_angles(roll, pitch, yaw)` uses the same convention but different floating evaluation; for bit-parity compose the three axis-angle quaternions in that order.
- `Rigid3d FromDictionary(LuaParameterDictionary*)`: `translation` sub-dict -> `GetArrayValuesAsDoubles`, `CHECK_EQ(3, size) "Need (x, y, z) for translation."`. `rotation` sub-dict: if `HasKey("w")` -> `Quaterniond(GetDouble w, x, y, z)`, `CHECK_NEAR(norm, 1., 1e-9)`; else array of 3 doubles `CHECK_EQ(3,..) "Need (roll, pitch, yaw) for rotation."` -> `RollPitchYaw`. (Keys read in order translation, rotation; quaternion keys w,x,y,z each exactly once.)

### rigid_transform_test_helpers.h
`ToEigen(Rigid2<T>) -> Eigen::Transform<T,2,Affine>` = `Translation * Rotation2D`; `ToEigen(Rigid3<T>) -> Transform<T,3,Affine>` = `Translation * Quaternion`. gmock `MATCHER_P2(IsNearly, rigid, epsilon)` = `ToEigen(arg).isApprox(ToEigen(rigid), epsilon)`. Eigen `isApprox` on the (N+1)x(N+1) matrices: `‖A − B‖_F ≤ eps · min(‖A‖_F, ‖B‖_F)` (**relative Frobenius**). Port this helper exactly (homogeneous matrices including bottom row [0..0 1]).

## 18. transform/transform.{h,cc}

| Function | Semantics |
|---|---|
| `F GetAngle(const Rigid3<F>&)` | `2 * atan2(‖q.vec‖, |q.w|)` -> angle in **[0, π]**. |
| `T GetYaw(const Eigen::Quaternion<T>&)` | `d = q * UnitX; return atan2(d.y, d.x)` (unqualified `atan2` -> ADL, works for ceres Jets). Yaw of ZYX decomposition, in (-π, π]. |
| `T GetYaw(const Rigid3<T>&)` | on rotation. |
| `Matrix<T,3,1> RotationQuaternionToAngleAxisVector(const Quaternion<T>&)` | normalize; if `w < 0` negate all 4 coeffs (choose shorter rotation); `angle = 2*atan2(‖vec‖, w)`; `scale = angle < 1e-7 ? 2 : angle / sin(angle/2)`; return `scale * vec`. |
| `Quaternion<T> AngleAxisVectorToRotationQuaternion(const Matrix<T,3,1>& aa)` | `scale = 0.5, w = 1`; if `aa.squaredNorm() > 1e-8` (i.e. ‖aa‖ > 1e-4): `n = ‖aa‖, scale = sin(n/2)/n, w = cos(n/2)`; return `(w, scale*aa)`. Not normalized in the linearized branch (norm slightly > 1). Note different cutoffs (1e-7 on angle vs 1e-8 on squared norm). |
| `Rigid2<T> Project2D(const Rigid3<T>&)` | `(t.head<2>(), GetYaw(t))`. |
| `Rigid3<T> Embed3D(const Rigid2<T>&)` | `({x, y, 0}, AngleAxis(angle, UnitZ))`. |
| `Rigid2d ToRigid2(const proto::Rigid2d&)` | |
| `ToEigen` overloads | `proto::Vector2d -> Vector2d`, `Vector3f -> Vector3f`, `Vector4f -> Vector4f (x,y,z,t)`, `Vector3d -> Vector3d`, `Quaterniond -> Quaterniond(w,x,y,z)` (no normalization). Note: there is **no** `ToEigen(proto::Quaternionf)` / `Vector2f`. |
| `ToProto` overloads | `Rigid2d`, `Rigid2f` (rotation = raw angle), `Rigid3d`, `Rigid3f`, `Vector2d`, `Vector3f`, `Vector4f` (t = w()), `Vector3d`, `Quaternionf`, `Quaterniond`. |
| `Rigid3d ToRigid3(const proto::Rigid3d&)` | no normalization. |
Note `transform.h` and `rigid_transform_test_helpers.h` both define `ToEigen` overloads (proto vs Rigid) — in Rust use distinct names (`From` impls).

Templates over T are instantiated with `double`, `float`, and `ceres::Jet` (mapping cost functions call `RotationQuaternionToAngleAxisVector`, `GetYaw`, etc.). Rust: generic over `T: RealField + Copy` (nalgebra) so an autodiff dual type can be plugged in; avoid `UnitQuaternion` in generic Jet code if the dual type can't satisfy its bounds.

## 19. transform/timestamped_transform.{h,cc}

`struct TimestampedTransform { common::Time time; Rigid3d transform; };`
- `FromProto(const proto::TimestampedTransform&)` = `{FromUniversal(time), ToRigid3(transform)}`; `ToProto` inverse.
- `TimestampedTransform Interpolate(const TT& start, const TT& end, Time time)`: `CHECK_LE(start.time, time)`, `CHECK_GE(end.time, time)`; `factor = ToSeconds(time - start.time) / ToSeconds(end.time - start.time)` (start==end -> 0/0 NaN); translation linear lerp `s + (e - s) * factor`; rotation `Quaterniond(start.q).slerp(factor, end.q)`.
- **Eigen slerp** (must replicate): `d = a·b; absD = |d|; if absD >= 1 - ε (ε = machine epsilon) { s0 = 1-t; s1 = t } else { θ = acos(absD); s0 = sin((1-t)θ)/sin θ; s1 = sin(tθ)/sin θ }; if d < 0 { s1 = -s1 }; return s0*a + s1*b` (coeff-wise, **shortest path**, no final normalization). nalgebra `UnitQuaternion::slerp` panics for antipodal inputs and uses different thresholds — write a custom `eigen_slerp`.

## 20. transform/transform_interpolation_buffer.{h,cc}

`constexpr size_t kUnlimitedBufferSize = std::numeric_limits<size_t>::max();`
`class TransformInterpolationBuffer { default ctor; explicit (const mapping::proto::Trajectory&); void SetSizeLimit(size_t); void Push(Time, const Rigid3d&); void Clear(); bool Has(Time) const; Rigid3d Lookup(Time) const; Time earliest_time() const; Time latest_time() const; bool empty() const; size_t size_limit() const; size_t size() const; private: void RemoveOldTransformsIfNeeded(); std::deque<TimestampedTransform> timestamped_transforms_; size_t buffer_size_limit_ = kUnlimitedBufferSize; }`
- Trajectory ctor: for each `node` push `(FromUniversal(node.timestamp()), ToRigid3(node.pose()))`.
- `Push`: if non-empty `CHECK_GE(time, latest_time()) "New transform is older than latest."` (equal timestamps allowed); push_back; trim.
- `SetSizeLimit`: set + trim. Trim = `while size > limit: pop_front` (limit 0 -> always empty).
- `Has(t)`: non-empty && `earliest <= t <= latest` (inclusive).
- `Lookup(t)`: `CHECK(Has(t)) "Missing transform for: " << t`; `end = lower_bound(by time < t)` (first element with time >= t); if `end.time == t` return it exactly (first of duplicates); else `Interpolate(*prev(end), *end, t).transform`.
- `earliest_time()/latest_time()`: `CHECK(!empty()) "Empty buffer."` (header comment claims "returns 0 if empty" — wrong; it aborts).
Rust: `VecDeque<TimestampedTransform>` + `partition_point(|x| x.time < t)`; `usize::MAX` for unlimited. Not thread-safe; single owner.

## 21. metrics/*

| File | API |
|---|---|
| counter.h/.cc | `class Counter { static Counter* Null(); virtual void Increment() = 0; virtual void Increment(double by_value) = 0; }`; `NullCounter` (no-ops) as function-local static. |
| gauge.h/.cc | `class Gauge { static Gauge* Null(); Increment(); Increment(double); Decrement(); Decrement(double); Set(double); }`; `NullGauge`. |
| histogram.h/.cc | `class Histogram { using BucketBoundaries = std::vector<double>; static Histogram* Null(); static BucketBoundaries FixedWidth(double width, int num_finite_buckets); static BucketBoundaries ScaledPowersOf(double base, double scale_factor, double max_value); virtual void Observe(double) = 0; }` |
| family_factory.h | `template<M> class Family { static Family<M>* Null(); virtual M* Add(const std::map<std::string,std::string>& labels) = 0; }`; `template<M> class NullFamily : Family<M> { Add -> M::Null(); }`; `class FamilyFactory { virtual Family<Counter>* NewCounterFamily(name, description) = 0; virtual Family<Gauge>* NewGaugeFamily(name, description) = 0; virtual Family<Histogram>* NewHistogramFamily(name, description, const Histogram::BucketBoundaries&) = 0; }` |
| register.h/.cc | `void RegisterAllMetrics(FamilyFactory* registry)` -> calls `RegisterMetrics(registry)` of `ConstraintBuilder2D`, `ConstraintBuilder3D`, `GlobalTrajectoryBuilderRegisterMetrics`, `LocalTrajectoryBuilder2D`, `LocalTrajectoryBuilder3D`, `PoseGraph2D`, `PoseGraph3D`, `sensor::TrajectoryCollator` (in that order). |

- `FixedWidth(w, n)`: `[w, 2w, ..., n·w]` by repeated addition (`boundary += width`, accumulates FP error — replicate by addition, not multiplication).
- `ScaledPowersOf(base, scale, max)`: `CHECK_GT(base,1)`, `CHECK_GT(scale,0)`; `b = scale; while (b < max) { push(b); b *= base; }` (max exclusive).
- Ownership: raw pointers owned by the concrete registry (e.g. Prometheus bridge in cartographer_ros / cloud), must outlive users. Pattern in clients: file-level `static auto* kFoo = metrics::Gauge::Null();` reassigned inside `RegisterMetrics(FamilyFactory*)` via `family->Add({{"label","value"}})`. Implementations must be thread-safe (called from worker threads). No upstream implementation here other than Null.
Rust: traits `Counter`, `Gauge`, `Histogram`, `Family<M: ?Sized>`, `FamilyFactory`, all `Send + Sync`; return `Arc<dyn Counter>` etc. (or `&'static dyn` via `Box::leak`). Null objects as `static NULL_COUNTER: NullCounter = NullCounter;`. Global metric slots: `static K_FOO: RwLock<Arc<dyn Gauge>>` / `OnceLock` / `arc_swap::ArcSwap` initialized to Null. Overloads `Increment()`/`Increment(double)` -> `increment()` + `increment_by(f64)`. Implement `RegisterAllMetrics` in the top-level crate (after mapping/sensor) to avoid a dependency cycle (metrics is a leaf crate otherwise).

---

## 22. Tests (every TEST / TEST_F)

### common/configuration_files_test.cc (needs mapping port)
| Test | Checks |
|---|---|
| ConfigurationFilesTest.ValidateMapBuilderOptions | Lua `include "map_builder.lua"; MAP_BUILDER.use_trajectory_builder_2d = true; return MAP_BUILDER` with `ConfigurationFileResolver({kSourceDirectory + "/configuration_files"})`; `CreateMapBuilderOptions` runs without fatal (implies every key used exactly once on dict destruction). |
| ConfigurationFilesTest.ValidateTrajectoryBuilderOptions | `include "trajectory_builder.lua"; TRAJECTORY_BUILDER.trajectory_builder_2d.use_imu_data = false; return TRAJECTORY_BUILDER`; `CreateTrajectoryBuilderOptions` without fatal. |

### common/fixed_ratio_sampler_test.cc
| Test | Checks |
|---|---|
| FixedRatioSamplerTest.AlwaysTrue | ratio 1.0: 100 pulses all true. |
| AlwaysFalse | ratio 0.0: 100 pulses all false. |
| NonSensicalRatio | death on 2.0 and -0.1, regex "ratio". |
| SometimesTrue | ratio 0.5: pulse i true iff i even (i=0..99). |
| FirstPulseIsTrue | ratio 1e-20: first true, next 100 false. |

### common/lua_parameter_dictionary_test.cc (fixture `LuaParameterDictionaryTest`, helper `ReferenceAllKeysAsIntegers` = GetInt every key; `MakeNonReferenceCounted`)
| Test | Input | Checks |
|---|---|---|
| GetInt | `return { blah = 100 }` | 100 |
| GetString | `return { blah = 'is_a_string' }\n` | "is_a_string" |
| GetDouble | `{ blah = 3.1415 }` | DOUBLE_EQ 3.1415 |
| GetBoolTrue / GetBoolFalse | `{ blah = true/false }` | true / false |
| GetDictionary | `{ blah = { blue = 100, red = 200 }, fasel = 10 }` | sub keys sorted == [blue, red], HasKey both, ints 100/200; `GetString("fasel") == "10"` (number->string coercion) |
| GetKeys | `{ blah = 100, blah1 = 200}` | 2 keys sorted [blah, blah1]; then references all (so dtor passes) |
| NonExistingKey | `{ blah = 100 }` | death `GetInt("blah_fasel")` regex `"Key.* not in dictionary."` |
| UintNegative | `{ blah = -100}` | death `GetNonNegativeInt` regex `".*-100 is negative."` (death runs in a forked child, so parent dict still needs keys referenced afterwards) |
| ToString | mixed table (ceta subtable, 1234.456786, ±math.huge, 123, 'hello', true, false) | exact golden string (sorted keys, `%f` numbers e.g. `123.000000`, quoted strings, nested indent 2 spaces, trailing commas); re-parse `"return " + ToString()` and read every value back (NEAR 1e-3 for fasel, isinf for huge) on both dicts |
| ToStringForArrays | non-ref-counted `{ "blub", 3, 3.1, foo = "ups" }` | golden `{\n  "blub",\n  3.000000,\n  3.100000,\n  foo = "ups",\n}` |
| GetArrayValuesAsStrings | `{ 'a','b','c' }` | GetKeys empty; [a,b,c] |
| GetArrayValuesAsDoubles | `{ 1., 2., 3. }` | GetKeys empty; NEAR 1e-7 |
| GetArrayValuesAsDictionaries | `{ { a = 1 }, { b = 3 } }` | 2 dicts; GetInt a=1, b=3 |
| TestChooseTrue / TestChoseFalse | `choose(true/false, 1, 0)` | 1 / 0 |
| TestChooseInvalidArgument | `choose('truish', 1, 0)` | death "condition is not a boolean value." |

### common/math_test.cc
| Test | Checks |
|---|---|
| MathTest.testPower | Power(0,42)=0, (0,0)=1, (1,0)=1, (1,42)=1, (2,2)=4 (exact) |
| testPow2 | 0,1,4,49 for 0,1,2,7 |
| testDeg2rad | DegToRad(180)≈π (1e-9); DegToRad(360-1e-9)≈2π (1e-6) |
| testRad2deg | RadToDeg(π)≈180 (1e-9); RadToDeg(2π-1e-9)≈360 (1e-6) |
| testNormalizeAngleDifference | 0->0, π->π, -π->-π, 2π->0, 5π->π, -5π->-π (1e-9) |

### common/task_test.cc (FakeThreadPool: synchronous, `RunNext()` executes front of queue, `IsEmpty()`; gmock `MockCallback::Run`)
| Test | Checks |
|---|---|
| TaskTest.RunTask | NEW before schedule; DEPENDENCIES_COMPLETED after; callback once; COMPLETED; queue empty. |
| RunTaskWithDependency | b depends on a; b DISPATCHED, a DEPENDENCIES_COMPLETED; after RunNext b DEPENDENCIES_COMPLETED; after 2nd both COMPLETED; callbacks in sequence a then b. |
| RunTaskWithTwoDependency | a->b->d, c->d; states: b,d DISPATCHED; after run a: a COMPLETED, b,c DEPENDENCIES_COMPLETED; after 2 more: b,c COMPLETED, d DEPENDENCIES_COMPLETED; then d COMPLETED. |
| RunWithCompletedDependency | b adds dependency on already-COMPLETED a: b stays NEW until scheduled, then immediately DEPENDENCIES_COMPLETED; runs. |

### common/thread_pool_test.cc (Receiver: mutex-protected vector, `WaitForNumberSequence` waits until size ≥ expected then EXPECT_EQ)
| Test | Checks |
|---|---|
| ThreadPoolTest.RunTask | pool(1), one task -> {1}. |
| ManyTasks | 5 rounds × pool(3), 10 tasks -> ten 1s. |
| RunWithDependency | pool(2), task_2 depends on task_1 -> {1,2}. |
| RunWithOutOfScopeDependency | weak handle goes out of scope before scheduling dependent -> still {1,2}. |
| ManyDependencies | 5 rounds × pool(5), task with 10 no-op deps -> {1}. |
| ManyDependants | 5 rounds × pool(5), 10 tasks depending on one -> ten 1s. |
| RunWithMultipleDependencies | diamond 1 -> {2a,2b} -> 3 (3 also directly on 1) -> {1,2,2,3}. |
| RunWithFinishedDependency | dependency added while task_1 possibly running/done; wait {1} then schedule task_2 -> {1,2}. |

### common/internal/blocking_queue_test.cc
| Test | Checks |
|---|---|
| BlockingQueueTest.testPushPeekPop | unique_ptr<int>: sizes 1,2; Peek=42 without removing; Pop 42 then 24; Peek on empty = nullptr. |
| testPushPopSharedPtr | shared_ptr FIFO 42, 24. |
| testPopWithTimeout | empty, 150 ms -> nullptr. |
| testPushWithTimeout | capacity 1: first true, second false (150 ms); pop 42; size 0. |
| testPushWithTimeoutInfinteQueue | infinite: both pushes true; FIFO. |
| testBlockingPop | thread blocks in Pop; main sleeps 100 ms then pushes 42; join; got 42, size 0. |
| testBlockingPopWithTimeout | same with PopWithTimeout(2500 ms). |

### common/internal/rate_timer_test.cc
| Test | Checks |
|---|---|
| RateTimerTest.ComputeRate | window 1 s, real steady clock, 100 pulses data time from UTS 42 step 0.1 s (all within 1 s wall, all kept) -> rate ≈ 10 (1e-3). |
| ComputeWallTimeRateRatio | SimulatedClock (static time_point advanced 0.05 s per pulse), data step 0.1 s -> ratio ≈ 2 (1e-3). |

### transform/rigid_transform_test.cc (TYPED_TEST over float, double; `eps = numeric_limits<T>::epsilon()`; random via `std::mt19937(42)`, `uniform_real_distribution(-1,1)`; Rigid2: x,y = 0.7·u, θ = 0.2·u; Rigid3: xyz = 0.7·u, angle-axis = 0.7·u via AngleAxisVectorToRotationQuaternion)
| Test | Checks (IsNearly with eps) |
|---|---|
| Identity2DTest | pose·Identity, Identity·pose (both default ctor and `Identity()`) ≈ pose |
| Inverse2DTest | inv·pose ≈ I, pose·inv ≈ I |
| Identity3DTest | same for Rigid3 |
| Inverse3DTest | same for Rigid3 |
Exact mt19937/uniform_real_distribution sequences aren't needed (properties hold for any sample); use a seeded `rand_chacha`/`StdRng`. Tolerance is relative Frobenius (isApprox) with machine epsilon — for f32 this is tight; keep the exact isApprox definition.

### transform/timestamped_transform_test.cc
| Test | Checks |
|---|---|
| TimestampedTransformTest.ToProtoAndBack | time UTS 12345678, Rigid3d(t=(1,2,3), q=(1,2,3,4).normalized()) -> proto -> back: time equal, transform IsNearly 1e-6. |

### transform/transform_test.cc
| Test | Checks |
|---|---|
| TransformTest.GetAngle | mt19937(42); 100 samples angle∈[0,π) float, random axis normalized: `GetAngle(Rigid3f::Rotation(AngleAxisVectorToRotationQuaternion(angle*axis))) ≈ angle` (1e-6f). |
| GetYaw | Z rotation 1.2345 -> 1.2345; inverse -> -1.2345 (1e-9). |
| GetYawAxisOrdering | Rz(1.2345)·Ry(0.4321)·Rx(0.6789) -> yaw 1.2345 (1e-9). |
| Embed3D | rigid3d = T(1,2,3)·R(q(1,2,3,4).normalized()); `rigid3d * Translation(1,2,0)` IsNearly `rigid3d * Embed3D(Rigid2d({1,2},0))` (1e-9). |

### transform/transform_interpolation_buffer_test.cc
| Test | Checks |
|---|---|
| TransformInterpolationBufferTest.testHas | empty Has(50)=false; after push 50: Has 25 F, 50 T, 75 F; after push 100: 25 F, 50/75/100 T, 125 F; earliest 50, latest 100. |
| testLookup | push (50, I), (100, T(10,10,10)·Rz(2)); Lookup(75) IsNearly T(5,5,5)·Rz(1) (1e-6). |
| testLookupSingleTransform | single push at 75; Lookup(75) ≈ I (1e-6). |
| testSetSizeLimit | default limit == kUnlimitedBufferSize; push 0,1,2 -> size 3; SetSizeLimit(2) -> size 2, Has(0) false; push 3 -> size 2, Has(1) false. |

metrics: **no tests**.

---

## 23. Consolidated Rust type mapping

| C++ | Rust |
|---|---|
| `int64`, etc. | `i64`, ... |
| `common::Time` / `Duration` | newtypes over `i64` (100 ns ticks), `Ord`, arithmetic ops |
| `Eigen::Vector{2,3,4}{d,f}` | `nalgebra::Vector{2,3,4}<f64/f32>` |
| `Eigen::Quaternion<T>` | `nalgebra::Quaternion<T>` (raw, may be non-unit) — wrap unit-assuming ops in own helpers; or `UnitQuaternion::new_unchecked` where Eigen assumes unit |
| `Eigen::Rotation2D<T>` | plain angle `T` (NOT `UnitComplex`/`Rotation2`, which lose the raw unwrapped angle) |
| `Eigen::AngleAxis<T>` | `(T, Vector3<T>)` or `nalgebra::Unit<Vector3>` + angle; convert with the Eigen formula |
| `Eigen::Transform<...,Affine>` (tests) | `nalgebra::Matrix3/Matrix4` homogeneous |
| `Rigid2<T>`, `Rigid3<T>` | `struct Rigid2<T> { translation: Vector2<T>, angle: T }`, `struct Rigid3<T> { translation: Vector3<T>, rotation: Quaternion<T> }`; impl `Mul<Rigid>`, `Mul<Vector>`, `inverse()`, `Display` = DebugString. Don't reuse `nalgebra::Isometry3` (its `Mul` doesn't renormalize the way Eigen code does and requires `UnitQuaternion`). |
| ceres `Jet` templates | generic `T: RealField + Copy` (autodiff dual numbers) |
| protos | `prost` |
| `absl::Mutex` + `Await(Condition)` | `std::sync::Mutex` + `Condvar` (`wait_while`, `wait_timeout_while`) / `parking_lot` |
| `std::shared_ptr<Task>` / `weak_ptr` | `Arc<Task>` / `Weak<Task>` |
| `std::function<void()>` | `Box<dyn FnOnce() + Send>` |
| `std::unique_ptr<FileResolver>` shared as `shared_ptr` | `Rc<dyn FileResolver>` (or `Arc` + `Send + Sync`) |
| Lua 5.2 C API | `mlua` (`lua52`, `vendored`) |
| boost gzip | `flate2` (`Compression::fast()`) |
| glog `LOG/CHECK` | `log` crate + `assert!`/`panic!` with identical messages |
| gflags | `clap` |
| `std::steady_clock` | `std::time::Instant` (abstract behind a `Clock` trait for RateTimer) |

## 24. Pitfall checklist
1. Rigid3 `*` renormalizes; Rigid2 `*` sums raw angles (no wrap); `Rigid2::rotation().angle()` can exceed π — use `normalized_angle()` when needed.
2. Rigid3 ctors and `ToRigid3` don't normalize; default proto quaternion is (0,0,0,0).
3. Quaternion arg order: Eigen/nalgebra ctor (w,x,y,z), storage [x,y,z,w], proto fields x=1..w=4, `FromArrays` [w,x,y,z], ceres arrays & `QuaternionProduct` [w,x,y,z], DebugString prints w first.
4. Slerp: replicate Eigen (shortest path, linear fallback at |d| ≥ 1-ε, no normalization).
5. `GetAngle` uses |w| -> [0, π]; `RotationQuaternionToAngleAxisVector` flips to w ≥ 0; cutoff 1e-7 (angle) vs `AngleAxisVectorToRotationQuaternion` cutoff 1e-8 (squared norm).
6. `NormalizeAngleDifference` inclusive bounds, port loops literally.
7. `FromSeconds` truncates toward zero; `ToSeconds` divides by 1e7.
8. Lua: exactly-once reference counting per string key, array access not counted, sub-dicts inherit mode, `GetString` accepts numbers, `GetKeys` excludes numeric-looking keys, `include` twice is fatal, `ToString` formats numbers with `%f`.
9. Task lock order dependency -> dependent -> pool; weak handle expires only after execution; `AddDependency` on an expired weak = no-op.
10. `TransformInterpolationBuffer::earliest_time/latest_time` abort on empty despite the comment; equal timestamps permitted on Push.
11. `FastGzipString` appends to output.
12. `metrics::RegisterAllMetrics` depends on mapping/sensor — place it in a top-level crate.

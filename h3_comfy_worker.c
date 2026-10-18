/* Persistent h3.c worker for the ComfyUI MiniMax H3 chain node.
 *
 * One request per stdin line, one JSON event per stdout line. A single h3_ctx
 * lives for the whole process so h3.c can keep its prepared DiT, conditioning
 * and video decoder caches between clips. Diagnostics go to stderr. SIGUSR1
 * cancels the running generation through the h3.c callbacks. */
#include "h3.h"

#include <errno.h>
#include <limits.h>
#include <mach/mach.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <zlib.h>

#ifndef H3_SOURCE_DIR
#error "build with make H3_DIR=/path/to/h3.c"
#endif

#define MAX_FIELDS 64
#define MAX_PHASES 32

typedef struct {
    char *key;
    char type; /* s string, n number, b boolean, z null */
    char *text;
} field;

typedef struct {
    field fields[MAX_FIELDS];
    int count;
    char error[256];
} request;

typedef struct {
    char name[48];
    double seconds;
} phase_time;

typedef struct {
    const char *last_frame;
    int last_frame_written;
    char error[512];
    double last_event;
    phase_time phases[MAX_PHASES];
    int phase_count;
} job;

static FILE *out;
static pid_t parent;
static volatile sig_atomic_t cancel_requested;

static void on_signal(int signal_number) {
    (void)signal_number;
    cancel_requested = 1;
}

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

/* ---- JSON input: one flat object of strings, numbers, booleans and null ---- */

static char *skip_ws(char *p) {
    while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
    return p;
}

static int hex4(const char *p, unsigned *value) {
    *value = 0;
    for (int i = 0; i < 4; i++) {
        char c = p[i];
        unsigned digit;
        if (c >= '0' && c <= '9') digit = (unsigned)(c - '0');
        else if (c >= 'a' && c <= 'f') digit = (unsigned)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') digit = (unsigned)(c - 'A' + 10);
        else return 0;
        *value = *value << 4 | digit;
    }
    return 1;
}

static void put_utf8(char **w, unsigned cp) {
    char *o = *w;
    if (cp < 0x80) {
        *o++ = (char)cp;
    } else if (cp < 0x800) {
        *o++ = (char)(0xC0 | cp >> 6);
        *o++ = (char)(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        *o++ = (char)(0xE0 | cp >> 12);
        *o++ = (char)(0x80 | (cp >> 6 & 0x3F));
        *o++ = (char)(0x80 | (cp & 0x3F));
    } else {
        *o++ = (char)(0xF0 | cp >> 18);
        *o++ = (char)(0x80 | (cp >> 12 & 0x3F));
        *o++ = (char)(0x80 | (cp >> 6 & 0x3F));
        *o++ = (char)(0x80 | (cp & 0x3F));
    }
    *w = o;
}

/* Decodes the string starting at the opening quote in place. Returns the
 * character after the closing quote, or NULL. Escapes never grow, so the
 * decoded text always fits. */
static char *parse_string(char *p, char **value) {
    char *w = ++p;
    *value = w;
    while (*p != '"') {
        if (!*p || (unsigned char)*p < 0x20) return NULL;
        if (*p != '\\') {
            *w++ = *p++;
            continue;
        }
        p++;
        switch (*p++) {
        case '"': *w++ = '"'; break;
        case '\\': *w++ = '\\'; break;
        case '/': *w++ = '/'; break;
        case 'b': *w++ = '\b'; break;
        case 'f': *w++ = '\f'; break;
        case 'n': *w++ = '\n'; break;
        case 'r': *w++ = '\r'; break;
        case 't': *w++ = '\t'; break;
        case 'u': {
            unsigned cp, low;
            if (!hex4(p, &cp)) return NULL;
            p += 4;
            if (cp >= 0xD800 && cp < 0xDC00) {
                if (p[0] != '\\' || p[1] != 'u' || !hex4(p + 2, &low) ||
                    low < 0xDC00 || low > 0xDFFF) return NULL;
                p += 6;
                cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
            } else if ((cp >= 0xDC00 && cp < 0xE000) || cp == 0) {
                return NULL;
            }
            put_utf8(&w, cp);
            break;
        }
        default:
            return NULL;
        }
    }
    *w = '\0';
    return p + 1;
}

static int parse_request(char *p, request *r) {
    r->count = 0;
    r->error[0] = '\0';
    p = skip_ws(p);
    if (*p++ != '{') return 0;
    p = skip_ws(p);
    if (*p == '}') return *skip_ws(p + 1) == '\0';
    for (;;) {
        if (r->count == MAX_FIELDS) return 0;
        field *f = &r->fields[r->count++];
        p = skip_ws(p);
        if (*p != '"' || !(p = parse_string(p, &f->key))) return 0;
        p = skip_ws(p);
        if (*p++ != ':') return 0;
        p = skip_ws(p);
        if (*p == '"') {
            f->type = 's';
            if (!(p = parse_string(p, &f->text))) return 0;
        } else if (!strncmp(p, "true", 4)) {
            f->type = 'b'; f->text = "1"; p += 4;
        } else if (!strncmp(p, "false", 5)) {
            f->type = 'b'; f->text = "0"; p += 5;
        } else if (!strncmp(p, "null", 4)) {
            f->type = 'z'; f->text = NULL; p += 4;
        } else {
            f->type = 'n';
            f->text = p;
            p += strspn(p, "+-0123456789.eE");
            if (p == f->text) return 0;
        }
        char *end = p;
        p = skip_ws(p);
        char c = *p;
        if (f->type == 'n') *end = '\0';
        if (c == '}') return *skip_ws(p + 1) == '\0';
        if (c != ',') return 0;
        p++;
    }
}

static field *find(request *r, const char *key) {
    for (int i = r->count - 1; i >= 0; i--)
        if (!strcmp(r->fields[i].key, key)) return &r->fields[i];
    return NULL;
}

/* Missing, null and empty strings all mean "not set". */
static const char *str_field(request *r, const char *key) {
    field *f = find(r, key);
    if (!f || f->type == 'z') return NULL;
    if (f->type != 's') {
        snprintf(r->error, sizeof(r->error), "%s must be a string", key);
        return NULL;
    }
    return *f->text ? f->text : NULL;
}

static int int_field(request *r, const char *key, int fallback) {
    field *f = find(r, key);
    if (!f || f->type == 'z') return fallback;
    if (f->type == 'b') return f->text[0] == '1';
    char *end;
    errno = 0;
    long value = f->type == 'n' ? strtol(f->text, &end, 10) : 0;
    if (f->type != 'n' || *end || errno || value < INT_MIN || value > INT_MAX) {
        snprintf(r->error, sizeof(r->error), "%s must be an integer", key);
        return fallback;
    }
    return (int)value;
}

static uint64_t u64_field(request *r, const char *key, uint64_t fallback) {
    field *f = find(r, key);
    if (!f || f->type == 'z') return fallback;
    char *end;
    errno = 0;
    unsigned long long value = f->type == 'n' && f->text[0] != '-' ?
        strtoull(f->text, &end, 10) : 0;
    if (f->type != 'n' || f->text[0] == '-' || *end || errno) {
        snprintf(r->error, sizeof(r->error), "%s must be an unsigned integer", key);
        return fallback;
    }
    return (uint64_t)value;
}

/* ---- JSON output ---- */

static void put_string(const char *s) {
    fputc('"', out);
    for (; s && *s; s++) {
        unsigned char c = (unsigned char)*s;
        if (c == '"' || c == '\\') fprintf(out, "\\%c", c);
        else if (c == '\n') fputs("\\n", out);
        else if (c < 0x20) fprintf(out, "\\u%04x", c);
        else fputc(c, out);
    }
    fputc('"', out);
}

static void finish_event(void) {
    fputs("}\n", out);
    if (fflush(out) != 0) exit(1); /* the node side is gone */
}

static void emit_error(const char *message, int cancelled) {
    fputs("{\"event\":\"error\",\"message\":", out);
    put_string(message && *message ? message : "unknown h3 error");
    fprintf(out, ",\"cancelled\":%s", cancelled ? "true" : "false");
    finish_event();
}

/* ---- generation callbacks ---- */

static void account(job *j, const char *phase) {
    double t = now();
    phase_time *slot = NULL;
    for (int i = 0; i < j->phase_count && !slot; i++)
        if (!strcmp(j->phases[i].name, phase)) slot = &j->phases[i];
    if (!slot && j->phase_count < MAX_PHASES) {
        slot = &j->phases[j->phase_count++];
        snprintf(slot->name, sizeof(slot->name), "%s", phase);
        slot->seconds = 0.0;
    }
    if (slot) slot->seconds += t - j->last_event;
    j->last_event = t;
}

static int stop_requested(void) {
    return cancel_requested || getppid() != parent;
}

/* Progress is reported after the work it counts, so the time since the
 * previous event belongs to this event's phase. */
static int on_progress(const char *phase, int completed, int total, void *opaque) {
    account(opaque, phase);
    fputs("{\"event\":\"progress\",\"phase\":", out);
    put_string(phase);
    fprintf(out, ",\"completed\":%d,\"total\":%d", completed, total);
    finish_event();
    return stop_requested();
}

static void put_be32(FILE *f, uint32_t value) {
    fputc((int)(value >> 24), f);
    fputc((int)(value >> 16 & 0xFF), f);
    fputc((int)(value >> 8 & 0xFF), f);
    fputc((int)(value & 0xFF), f);
}

static void put_chunk(FILE *f, const char *type, const uint8_t *data, uint32_t length) {
    put_be32(f, length);
    fwrite(type, 1, 4, f);
    if (length) fwrite(data, 1, length, f);
    uLong crc = crc32(0L, (const Bytef *)type, 4);
    if (length) crc = crc32(crc, data, length);
    put_be32(f, (uint32_t)crc);
}

static int write_png(const char *path, const h3_frame *frame, char *error, size_t size) {
    size_t row = (size_t)frame->width * 3;
    size_t raw_size = (row + 1) * (size_t)frame->height;
    uLongf packed_size = compressBound((uLong)raw_size);
    uint8_t *raw = malloc(raw_size);
    uint8_t *packed = malloc(packed_size);
    int ok = raw && packed;
    if (ok) {
        for (int y = 0; y < frame->height; y++) {
            uint8_t *line = raw + (size_t)y * (row + 1);
            line[0] = 0; /* no filter */
            memcpy(line + 1, frame->rgb + (size_t)y * (size_t)frame->stride, row);
        }
        ok = compress2(packed, &packed_size, raw, (uLong)raw_size, Z_DEFAULT_COMPRESSION) == Z_OK;
    }
    FILE *f = ok ? fopen(path, "wb") : NULL;
    if (f) {
        static const uint8_t signature[8] = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n'};
        uint8_t header[13] = {
            (uint8_t)(frame->width >> 24), (uint8_t)(frame->width >> 16),
            (uint8_t)(frame->width >> 8), (uint8_t)frame->width,
            (uint8_t)(frame->height >> 24), (uint8_t)(frame->height >> 16),
            (uint8_t)(frame->height >> 8), (uint8_t)frame->height,
            8, 2, 0, 0, 0}; /* 8-bit RGB */
        fwrite(signature, 1, sizeof(signature), f);
        put_chunk(f, "IHDR", header, sizeof(header));
        put_chunk(f, "IDAT", packed, (uint32_t)packed_size);
        put_chunk(f, "IEND", NULL, 0);
        ok = !ferror(f);
        ok = fclose(f) == 0 && ok;
    }
    if (!f || !ok) snprintf(error, size, "cannot write last frame %s: %s", path,
                            ok ? strerror(errno) : "write failed");
    free(raw);
    free(packed);
    return f && ok;
}

/* h3.c has no progress callback for the video VAE decode and delivers the frames
 * once they are decoded: the first frame's event carries the whole decode time. */
static int on_frame(const h3_frame *frame, void *opaque) {
    job *j = opaque;
    if (frame->denoise_step >= 0) return stop_requested();
    if (frame->frame_index == 0 && on_progress("video VAE decode", frame->frame_count, frame->frame_count, j)) return 1;
    if (frame->frame_index == frame->frame_count - 1 && j->last_frame) {
        if (!write_png(j->last_frame, frame, j->error, sizeof(j->error))) return 1;
        j->last_frame_written = 1;
        return on_progress("last frame", 1, 1, j);
    }
    return stop_requested();
}

/* ---- commands ---- */

static uint64_t peak_footprint(void) {
    task_vm_info_data_t info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &count) != KERN_SUCCESS)
        return 0;
    return info.ledger_phys_footprint_peak;
}

static h3_ctx *ctx;
static char *loaded_dir;
static int load_count, generate_count;

static void load(request *r) {
    const char *model_dir = str_field(r, "model_dir");
    if (r->error[0] || !model_dir) {
        emit_error(r->error[0] ? r->error : "load requires model_dir", 0);
        return;
    }
    if (!ctx || strcmp(loaded_dir, model_dir)) {
        h3_free(ctx);
        free(loaded_dir);
        loaded_dir = NULL;
        load_count++;
        fprintf(stderr, "h3-worker: h3_load_dir #%d %s\n", load_count, model_dir);
        ctx = h3_load_dir(model_dir);
        if (!ctx) {
            emit_error(h3_last_error(NULL), 0);
            return;
        }
        h3_cache_set_enabled(ctx, 1);
        loaded_dir = strdup(model_dir);
    }
    const h3_device_info *device = h3_device(ctx);
    fputs("{\"event\":\"loaded\",\"model_dir\":", out);
    put_string(loaded_dir);
    fputs(",\"device\":", out);
    put_string(device->name);
    fprintf(out, ",\"metal4\":%s,\"load_count\":%d",
            device->metal4 ? "true" : "false", load_count);
    finish_event();
}

static void generate(request *r) {
    if (!ctx) {
        emit_error("no model loaded", 0);
        return;
    }
    const char *prompt = str_field(r, "prompt");
    job j;
    memset(&j, 0, sizeof(j));
    j.last_frame = str_field(r, "last_frame_output");
    h3_params params = H3_PARAMS_DEFAULT;
    params.width = int_field(r, "width", params.width);
    params.height = int_field(r, "height", params.height);
    params.frames = int_field(r, "frames", params.frames);
    params.steps = int_field(r, "steps", params.steps);
    params.seed = u64_field(r, "seed", params.seed);
    params.output_path = str_field(r, "output");
    params.first_frame = str_field(r, "first_frame");
    params.last_frame = str_field(r, "last_frame");
    params.denoise_reuse = int_field(r, "reuse", params.denoise_reuse);
    params.dit_layers = int_field(r, "layers", params.dit_layers);
    params.core_reuse = int_field(r, "core_reuse", params.core_reuse);
    params.token_reduction = int_field(r, "token_reduction", params.token_reduction);
    params.render_width = int_field(r, "render_width", params.render_width);
    params.render_height = int_field(r, "render_height", params.render_height);
    params.ssd_streaming = int_field(r, "ssd_streaming", params.ssd_streaming);
    /* Ref2VA references, in <Picture 1>, <Audio 1> order. */
    h3_reference references[2];
    const char *ref_image = str_field(r, "ref_image");
    const char *ref_audio = str_field(r, "ref_audio");
    if (ref_image) references[params.reference_count++] = (h3_reference){H3_REFERENCE_IMAGE, ref_image, NULL, 0};
    if (ref_audio) references[params.reference_count++] = (h3_reference){H3_REFERENCE_AUDIO, ref_audio, NULL, 0};
    if (params.reference_count) params.references = references;
    params.on_frame = on_frame;
    params.on_progress = on_progress;
    params.callback_opaque = &j;
    if (r->error[0]) {
        emit_error(r->error, 0);
        return;
    }

    cancel_requested = 0;
    generate_count++;
    fprintf(stderr, "h3-worker: h3_generate #%d %dx%d frames=%d steps=%d seed=%llu first=%s last=%s ref_image=%s ref_audio=%s\n",
            generate_count, params.width, params.height, params.frames, params.steps,
            (unsigned long long)params.seed,
            params.first_frame ? params.first_frame : "-",
            params.last_frame ? params.last_frame : "-",
            ref_image ? ref_image : "-", ref_audio ? ref_audio : "-");
    double start = now();
    j.last_event = start;
    h3_result *result = h3_generate(ctx, prompt ? prompt : "", &params);
    account(&j, "finish");
    if (!result || (j.last_frame && !j.last_frame_written)) {
        int cancelled = cancel_requested != 0;
        emit_error(j.error[0] ? j.error : result ? "h3 delivered no final frame" : h3_last_error(ctx),
                   cancelled);
        h3_result_free(result);
        return;
    }
    h3_cache_info cache;
    h3_cache_get_info(ctx, &cache);
    fputs("{\"event\":\"complete\",\"output\":", out);
    put_string(params.output_path);
    fputs(",\"last_frame\":", out);
    put_string(j.last_frame);
    fprintf(out, ",\"width\":%d,\"height\":%d,\"frames\":%d,\"fps\":%d,\"sample_rate\":%d",
            result->width, result->height, result->frames, result->fps, result->sample_rate);
    fprintf(out, ",\"seconds\":%.3f,\"generate_count\":%d,\"load_count\":%d",
            now() - start, generate_count, load_count);
    fprintf(out, ",\"cache\":{\"embedding_entries\":%zu,\"prepared_dit\":%d,\"video_decoder\":%d}",
            cache.embedding_entries, cache.prepared_dit, cache.video_decoder);
    fprintf(out, ",\"peak_footprint\":%llu,\"timings\":{", (unsigned long long)peak_footprint());
    for (int i = 0; i < j.phase_count; i++) {
        if (i) fputc(',', out);
        put_string(j.phases[i].name);
        fprintf(out, ":%.3f", j.phases[i].seconds);
    }
    fputc('}', out);
    finish_event();
    h3_result_free(result);
}

int main(void) {
    struct sigaction action;
    memset(&action, 0, sizeof(action));
    action.sa_handler = on_signal;
    action.sa_flags = SA_RESTART;
    sigaction(SIGUSR1, &action, NULL);
    signal(SIGPIPE, SIG_IGN);
    parent = getppid();

    /* Keep the protocol on a private copy of stdout; anything else that
     * writes to fd 1 ends up in the diagnostic stream. */
    out = fdopen(dup(STDOUT_FILENO), "w");
    if (!out || dup2(STDERR_FILENO, STDOUT_FILENO) < 0) return 1;

    /* h3.c compiles h3_shaders.metal from the working directory at runtime. */
    if (chdir(H3_SOURCE_DIR) != 0 || access("h3_shaders.metal", R_OK) != 0) {
        char message[PATH_MAX + 128];
        snprintf(message, sizeof(message), "cannot find h3_shaders.metal in the h3.c checkout %s: %s",
                 H3_SOURCE_DIR, strerror(errno));
        emit_error(message, 0);
        return 1;
    }
    fprintf(out, "{\"event\":\"ready\",\"pid\":%d,\"h3_version\":", (int)getpid());
    put_string(H3_VERSION);
    fputs(",\"h3_dir\":", out);
    put_string(H3_SOURCE_DIR);
    finish_event();

    char *line = NULL;
    size_t capacity = 0;
    request r;
    while (getline(&line, &capacity, stdin) > 0) {
        if (!parse_request(line, &r)) {
            emit_error("malformed request: expected one flat JSON object per line", 0);
            continue;
        }
        const char *command = str_field(&r, "command");
        if (command && !strcmp(command, "load")) load(&r);
        else if (command && !strcmp(command, "generate")) generate(&r);
        else if (command && !strcmp(command, "quit")) break;
        else emit_error("unknown command", 0);
    }
    free(line);
    h3_free(ctx);
    free(loaded_dir);
    return 0;
}

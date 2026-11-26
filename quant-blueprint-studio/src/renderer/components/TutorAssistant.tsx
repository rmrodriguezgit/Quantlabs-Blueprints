import { useEffect, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { BookOpen, LoaderCircle, MessageCircleQuestion, Plus, Send, Sparkles, Trash2, Workflow, X } from "lucide-react";
import { askTutor, type TutorAnswer, type TutorSource } from "../brain";
import { catalog, type Kind } from "../types/blueprint";
import { LlmStatus } from "./LlmStatus";

type Message = { role: "user"; text: string } | { role: "assistant"; result: TutorAnswer } | { role: "error"; text: string };
const KEY = "quantlabs-tutor";
const STARTERS = ["¿Qué es el ratio de Sharpe?", "¿Cómo evito la fuga de información?", "¿Qué herramienta uso para una red neuronal?", "¿Qué es la calibración?"];
const KIND_LABEL: Record<TutorSource["kind"], string> = { tool: "Herramienta", concept: "Concepto", example: "Ejemplo" };

function restore(): Message[] {
  try {
    const saved: unknown = JSON.parse(sessionStorage.getItem(KEY) ?? "[]");
    return Array.isArray(saved) ? (saved as Message[]).slice(-30) : [];
  } catch {
    return [];
  }
}

/** **negritas** y párrafos; el resto es texto (la respuesta es dato, nunca HTML). */
function Rich({ text }: { text: string }) {
  return (
    <>
      {text.split(/\n{2,}/).map((paragraph, i) => (
        <p key={i}>
          {paragraph.split(/(\*\*[^*]+\*\*)/).map(
            (part, j): ReactNode =>
              part.startsWith("**") && part.endsWith("**") ? (
                <strong key={j}>{part.slice(2, -2)}</strong>
              ) : (
                part.replace(/\[(tool|concept|example):[^\]]+\]/g, "").replace(/\s+([.,;])/g, "$1")
              ),
          )}
        </p>
      ))}
    </>
  );
}

/** Cerebro Tutor: resuelve dudas sobre herramientas, conceptos y ejemplos. Botón flotante abajo a la derecha. */
export function TutorAssistant({
  selectedKind,
  onAddTool,
  onOpenExample,
  onPlan,
}: {
  selectedKind?: Kind;
  onAddTool: (kind: Kind) => void;
  onOpenExample: (id: string) => void;
  onPlan: (question: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>(restore);
  const [draft, setDraft] = useState("");
  const [pending, setPending] = useState(false);
  const list = useRef<HTMLDivElement>(null);
  const input = useRef<HTMLTextAreaElement>(null);
  useEffect(() => {
    try {
      sessionStorage.setItem(KEY, JSON.stringify(messages.slice(-30)));
    } catch {
      /* sin almacenamiento */
    }
  }, [messages]);
  useEffect(() => {
    list.current?.scrollTo({ top: list.current.scrollHeight, behavior: "smooth" });
  }, [messages, pending, open]);
  useEffect(() => {
    if (open) input.current?.focus();
  }, [open]);

  const ask = async (question: string) => {
    const text = question.trim();
    if (!text || pending) return;
    setDraft("");
    setMessages((m) => [...m, { role: "user", text }]);
    setPending(true);
    try {
      const result = await askTutor(text, selectedKind);
      setMessages((m) => [...m, { role: "assistant", result }]);
    } catch (e) {
      setMessages((m) => [...m, { role: "error", text: e instanceof Error ? e.message : String(e) }]);
    } finally {
      setPending(false);
    }
  };

  const last = [...messages].reverse().find((m): m is Extract<Message, { role: "assistant" }> => m.role === "assistant");
  const suggestions = last?.result.follow_up.length
    ? last.result.follow_up
    : selectedKind
      ? [`¿Qué hace este nodo (${catalog[selectedKind].label})?`, `¿Cómo interpreto ${catalog[selectedKind].label}?`, STARTERS[0]]
      : STARTERS;

  const panel = open ? (
    <section
      className="tutor-panel"
      role="dialog"
      aria-label="Cerebro Tutor"
      onKeyDown={(e) => {
        if (e.key === "Escape") {
          e.stopPropagation();
          setOpen(false);
        }
      }}
    >
      <header>
        <span className="tutor-emblem">
          <MessageCircleQuestion size={16} />
        </span>
        <div>
          <strong>Cerebro Tutor</strong>
          <small>Dudas sobre herramientas, conceptos y ejemplos</small>
        </div>
        <LlmStatus compact />
        <button className="icon-button" title="Borrar la conversación" disabled={!messages.length || pending} onClick={() => setMessages([])}>
          <Trash2 size={14} />
        </button>
        <button className="icon-button" title="Cerrar (Esc)" onClick={() => setOpen(false)}>
          <X size={15} />
        </button>
      </header>
      <div className="tutor-messages" ref={list} aria-live="polite">
        {!messages.length && (
          <div className="tutor-empty">
            <Sparkles size={18} />
            <p>
              Pregunta qué hace una herramienta, qué significa un concepto o qué enseña un ejemplo. Responde con las fichas del catálogo, el glosario y la
              biblioteca{selectedKind ? `; «este nodo» es ${catalog[selectedKind].label}` : ""}.
            </p>
          </div>
        )}
        {messages.map((m, i) =>
          m.role === "user" ? (
            <div key={i} className="tutor-msg user">
              {m.text}
            </div>
          ) : m.role === "error" ? (
            <div key={i} className="tutor-msg error">
              No se pudo responder: {m.text}
            </div>
          ) : (
            <div key={i} className="tutor-msg assistant">
              <Rich text={m.result.answer} />
              {m.result.sources.length > 0 && (
                <div className="tutor-sources" aria-label="Fuentes">
                  {m.result.sources.map((s) => (
                    <span key={s.id} className={`tutor-source ${s.kind}`}>
                      {KIND_LABEL[s.kind]} · {s.title}
                    </span>
                  ))}
                </div>
              )}
              <div className="tutor-actions">
                {m.result.actions.map((a) =>
                  a.type === "add_tool" ? (
                    <button key={a.kind} disabled={!(a.kind in catalog)} onClick={() => onAddTool(a.kind as Kind)}>
                      <Plus size={12} /> Añadir {a.label}
                    </button>
                  ) : (
                    <button key={a.id} onClick={() => onOpenExample(a.id)}>
                      <BookOpen size={12} /> Abrir {a.label}
                    </button>
                  ),
                )}
                <button className="tutor-to-builder" title="Abre el Cerebro Constructor con esta pregunta como enunciado" onClick={() => onPlan(m.result.question)}>
                  <Workflow size={12} /> Pasar al Constructor
                </button>
              </div>
              <small className="tutor-engine">
                {m.result.engine === "ollama"
                  ? `Redactado por ${m.result.model} con las fichas citadas`
                  : "Compuesto con las fichas citadas (sin modelo de lenguaje)"}
                {m.result.fallback_reason && m.result.engine === "rules" ? " · modelo local no disponible" : ""}
              </small>
            </div>
          ),
        )}
        {pending && (
          <div className="tutor-msg assistant pending">
            <LoaderCircle className="spin" size={14} /> Buscando en las fichas…
          </div>
        )}
      </div>
      <div className="tutor-suggestions">
        {suggestions.slice(0, 3).map((s) => (
          <button key={s} disabled={pending} onClick={() => void ask(s)}>
            {s}
          </button>
        ))}
      </div>
      <form
        className="tutor-input"
        onSubmit={(e) => {
          e.preventDefault();
          void ask(draft);
        }}
      >
        <textarea
          ref={input}
          rows={2}
          value={draft}
          maxLength={1000}
          placeholder="Escribe tu pregunta…"
          aria-label="Pregunta para el Cerebro Tutor"
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              void ask(draft);
            }
          }}
        />
        <button type="submit" className="tutor-send" disabled={!draft.trim() || pending} aria-label="Enviar pregunta">
          <Send size={15} />
        </button>
      </form>
    </section>
  ) : null;

  return (
    <div className={`tutor ${open ? "open" : ""}`}>
      <button className="tutor-fab" aria-expanded={open} title="Cerebro Tutor: resuelve dudas" onClick={() => setOpen((v) => !v)}>
        <span className="tutor-emblem">
          <MessageCircleQuestion size={16} />
        </span>
        <span className="tutor-fab-text">
          <strong>Cerebro Tutor</strong>
          <small>{pending ? "Respondiendo…" : "Resuelve dudas"}</small>
        </span>
        {open && <X size={14} />}
      </button>
      {panel && createPortal(panel, document.body)}
    </div>
  );
}

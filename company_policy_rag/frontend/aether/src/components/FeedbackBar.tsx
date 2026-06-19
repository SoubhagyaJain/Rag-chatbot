import { ThumbsDown, ThumbsUp } from "lucide-react";
import { useState } from "react";
import { submitFeedback } from "../api/client";

interface Props {
  messageId?: string;
  question: string;
  answer: string;
  model: string;
  corpusScope: string;
  onToast: (msg: string) => void;
}

export function FeedbackBar({
  messageId,
  question,
  answer,
  model,
  corpusScope,
  onToast,
}: Props) {
  const [submitted, setSubmitted] = useState<1 | -1 | null>(null);
  const [commentOpen, setCommentOpen] = useState(false);
  const [comment, setComment] = useState("");

  const send = async (rating: 1 | -1) => {
    if (submitted) return;
    try {
      await submitFeedback({
        rating,
        question,
        answer,
        model,
        corpus_scope: corpusScope,
        message_id: messageId,
        comment: rating === -1 ? comment || undefined : undefined,
      });
      setSubmitted(rating);
      setCommentOpen(false);
      onToast(rating === 1 ? "Thanks for the feedback" : "Feedback recorded");
    } catch {
      onToast("Could not save feedback");
    }
  };

  return (
    <div className="mt-2 flex items-center gap-2 flex-wrap">
      <button
        disabled={submitted !== null}
        onClick={() => send(1)}
        className={`p-1.5 rounded-lg transition-colors ${
          submitted === 1 ? "text-emerald-500" : "text-white/40 hover:text-white/80 hover:bg-white/5"
        }`}
        aria-label="Helpful"
      >
        <ThumbsUp className="w-4 h-4" />
      </button>
      <button
        disabled={submitted !== null}
        onClick={() => {
          if (commentOpen) send(-1);
          else setCommentOpen(true);
        }}
        className={`p-1.5 rounded-lg transition-colors ${
          submitted === -1 ? "text-amber-500" : "text-white/40 hover:text-white/80 hover:bg-white/5"
        }`}
        aria-label="Not helpful"
      >
        <ThumbsDown className="w-4 h-4" />
      </button>
      {commentOpen && submitted === null && (
        <input
          value={comment}
          onChange={(e) => setComment(e.target.value)}
          placeholder="What was wrong?"
          className="flex-1 min-w-[140px] text-xs bg-white/10 border border-white/10 rounded-lg px-2 py-1 text-white placeholder-white/40 outline-none focus:ring-1 focus:ring-primary"
          onKeyDown={(e) => e.key === "Enter" && send(-1)}
        />
      )}
    </div>
  );
}
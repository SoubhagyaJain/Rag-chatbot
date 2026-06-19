export function HeroOrb() {
  return (
    <div className="flex flex-col items-center justify-center pt-8 pb-4">
      <div className="orb w-28 h-28 rounded-full bg-gradient-to-br from-primary via-highlight to-violet-300 shadow-[0_0_40px_rgba(168,85,247,0.35)] mb-8 relative">
        <div className="absolute inset-2 rounded-full bg-gradient-to-t from-primary/30 to-transparent" />
      </div>
      <h1 className="text-2xl font-semibold text-sidebar mb-2 tracking-tight">
        What would you like to explore today?
      </h1>
      <p className="text-sidebar/50 text-sm mb-8 text-center max-w-md">
        Ask anything about your company policies or the AI Agents guidebook
      </p>
    </div>
  );
}
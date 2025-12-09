import { type HTMLAttributes } from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils/cn";

const badgeVariants = cva(
  "inline-flex items-center rounded-md border px-2.5 py-0.5 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2",
  {
    variants: {
      variant: {
        default:
          "border-transparent bg-primary text-primary-foreground shadow hover:bg-primary/80",
        secondary:
          "border-transparent bg-secondary text-secondary-foreground hover:bg-secondary/80",
        destructive:
          "border-transparent bg-destructive text-destructive-foreground shadow hover:bg-destructive/80",
        success:
          "border-transparent bg-success text-success-foreground shadow hover:bg-success/80",
        warning:
          "border-transparent bg-warning text-warning-foreground shadow hover:bg-warning/80",
        outline: "text-foreground",
        recording:
          "border-transparent bg-recording text-recording-foreground shadow animate-recording-pulse",
        connected:
          "border-transparent bg-connected text-white shadow",
        disconnected:
          "border-transparent bg-disconnected text-white shadow",
        speaker1: "border-transparent bg-speaker-1 text-white shadow",
        speaker2: "border-transparent bg-speaker-2 text-white shadow",
        speaker3: "border-transparent bg-speaker-3 text-white shadow",
        speaker4: "border-transparent bg-speaker-4 text-white shadow",
        speaker5: "border-transparent bg-speaker-5 text-white shadow",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  },
);

export interface DDXBadgeProps
  extends HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

function DDXBadge({ className, variant, ...props }: DDXBadgeProps): React.ReactNode {
  return (
    <div className={cn(badgeVariants({ variant }), className)} {...props} />
  );
}

export { DDXBadge, badgeVariants };

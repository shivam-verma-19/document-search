import { NextResponse } from "next/server";

export function apiErrorResponse(
  status: number,
  code: string,
  message: string,
): NextResponse {
  return NextResponse.json(
    {
      error: {
        code,
        message,
        retryable: status >= 500 || status === 429,
      },
    },
    { status },
  );
}

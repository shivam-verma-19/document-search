import { NextRequest, NextResponse } from "next/server";
import { proxyToBackend } from "../../../../lib/backend-proxy";

type RouteContext = {
  params: Promise<{ path: string[] }>;
};

async function proxy(
  request: NextRequest,
  context: RouteContext,
): Promise<NextResponse> {
  const { path } = await context.params;
  return proxyToBackend(request, path);
}

export const GET = proxy;
export const HEAD = proxy;
export const POST = proxy;
export const PUT = proxy;
export const PATCH = proxy;
export const DELETE = proxy;

export async function OPTIONS() {
  return new NextResponse(null, { status: 204 });
}

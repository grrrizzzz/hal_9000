import { QueryClient } from "@tanstack/react-query";
import { ApiError } from "@/api/client";

export function makeQueryClient(): QueryClient {
  return new QueryClient({
    defaultOptions: {
      queries: {
        staleTime: 15_000,
        refetchOnWindowFocus: false,
        retry: (count, err) => {
          if (err instanceof ApiError && [401, 403, 404, 422, 501].includes(err.status)) return false;
          return count < 2;
        },
      },
      mutations: { retry: false },
    },
  });
}

See [discussion](https://github.com/iimez/inference-server/discussions/8) for more details.

### Task API

- `POST /tasks/text-completion`
- `POST /tasks/chat-completion` should continue to be stateless
- `GET /tasks/{task_id}`
- `DELETE /tasks/{task_id}`

### Thread API

- `POST /threads`
- `POST /threads/{thread_id}` mutate state without generating anything
- `POST /threads/{thread_id}/generate` to generate a new assistant message
- `GET /threads/{thread_id}`
- `DELETE /threads/{thread_id}`